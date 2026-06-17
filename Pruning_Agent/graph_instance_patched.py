#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
graph_instance_patched.py
=========================
Đây là phiên bản graph_instance.py đã tích hợp PruningAgent + FastGA.
Chỉ cần đặt pruning_agent.py cùng thư mục rồi chạy file này thay cho file gốc.

Thay đổi so với gốc:
  1. Import PruningAgent, FastGA từ pruning_agent.py
  2. Mỗi vòng stage, đồ thị được prune trước khi GA chạy
  3. GA được thay bằng FastGA (tương thích 100% về output)
  4. Kết quả GA được remap về index gốc trước khi ghi file
"""

import os
import random
import re
import json
import logging
from pruning_agent import PruningAgent, FastGA

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ─── Cấu hình PruningAgent ────────────────────────────────────────────────────
PRUNING_TARGET_NODES   = 150    # giảm đồ thị xuống ≤150 node trước khi GA
MIN_STAGE_COVERAGE     = 0.60   # giữ ≥60% node quan trọng của stage hiện tại
HUB_EXPANSION_HOPS     = 3      # BFS 3 hop từ hub node
PRUNING_VERBOSE        = True   # In log chi tiết
# ─────────────────────────────────────────────────────────────────────────────

# GA params (giữ nguyên như gốc, chỉ popsize tăng lên 20 là đủ sau prune)
CXPB, MUTPB, NGEN, popsize = 0.8, 0.4, 5, 20

stage_len = [58, 119, 94, 101]


class Dataloader:
    def __init__(self):
        self.regu_path     = r".\regulation_dic"
        self.tech_path     = r".\tech_dic"
        self.sub_graph_path = r".\4000_3_generated_data_new2_sub"

    def get_relation_list(self, data):
        relation_list = [[], [], [], []]
        i1 = 0
        for line in data:
            relation = []
            pattern = r'(\d+)\s+(\d+)\s+([A-Z]+)+(-\d+)'
            matches = re.findall(pattern, line)
            for match in matches:
                num1, num2, text, stage = match
                stage_true = int(stage[-1]) - 1
                relation.append(num1)
                relation.append(num2)
                relation.append(text)
                relation.append(i1)
                i1 += 1
                relation_list[stage_true].append(relation)
                relation = []
        return relation_list

    def get_entity_list(self, data):
        entity_list = []
        for line in data:
            pattern = r"^([A-Z]{2})\*?"
            matches = re.findall(pattern, line)
            for match in matches:
                entity_list.append(match)
        return entity_list

    def get_graph_info(self, whole_file_path):
        with open(whole_file_path, "r") as file:
            data = file.readlines()
        entity_list   = self.get_entity_list(data)
        relation_list = self.get_relation_list(data)
        return data, entity_list, relation_list

    def read_json(self, file_path):
        with open(file_path, "r") as file:
            return json.load(file)

    def load_regulation(self):
        dics = {}
        for regulation in os.listdir(self.regu_path):
            whole_path = os.path.join(self.regu_path, regulation)
            for s in [1, 2, 3, 4]:
                if f"stage{s}" in regulation:
                    dics[s] = self.read_json(whole_path)
        return dics[1], dics[2], dics[3], dics[4]

    def load_tech(self):
        dics = {}
        for tech in os.listdir(self.tech_path):
            whole_path = os.path.join(self.tech_path, tech)
            for s in [1, 2, 3, 4]:
                if f"stage{s}" in tech:
                    dics[s] = self.read_json(whole_path)
        return dics[1], dics[2], dics[3], dics[4]


def read_new(json_file):
    file = open(json_file, "r", encoding="utf-8")
    papers = []
    for line in file.readlines():
        dic = json.loads(line, strict=False)
        papers.append(dic)
    file.close()
    return papers


def make_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return 1


def find_target_relation(relation_list, key):
    for rela in relation_list:
        if rela[3] == key:
            return rela


if __name__ == "__main__":

    DL = Dataloader()
    (stage1_index_regu_dic, stage2_index_regu_dic,
     stage3_index_regu_dic, stage4_index_regu_dic) = DL.load_regulation()
    (stage1_index_tech_dic, stage2_index_tech_dic,
     stage3_index_tech_dic, stage4_index_tech_dic) = DL.load_tech()

    instance_lib = read_new(r".\instance_lib\technique-instance-lib-os-filter-add.json")
    new_instance_path = r".\4000_3_generated_data_new2_sub_instance_windows"
    make_directory(new_instance_path)
    os_type = "linux"

    # Khởi tạo PruningAgent một lần, tái dụng cho tất cả file
    pruning_agent = PruningAgent(
        target_nodes=PRUNING_TARGET_NODES,
        min_stage_coverage=MIN_STAGE_COVERAGE,
        hub_expansion_hops=HUB_EXPANSION_HOPS,
        verbose=PRUNING_VERBOSE,
    )

    for txt in os.listdir(DL.sub_graph_path):
        whole_file_path = os.path.join(DL.sub_graph_path, txt)
        new_file_path   = os.path.join(new_instance_path, txt)

        graph_data, entity_list, relation_list1 = DL.get_graph_info(whole_file_path)
        logger.info(f"\n{'='*50}\nProcessing: {txt} | nodes={len(entity_list)}")

        entity_instance_dic  = {i: [] for i in range(len(entity_list))}
        relation_instance_dic = {}
        sum_relation = sum(len(s) for s in relation_list1)
        for i in range(sum_relation):
            relation_instance_dic[i] = []

        for stage in range(1, 5):
            stage_index_regu_dic = DL.load_regulation()[stage - 1]
            stage_index_tech_dic = DL.load_tech()[stage - 1]
            stage_regu_len       = stage_len[stage - 1]

            # ── PRUNING: cắt đồ thị trước khi GA ──────────────────────────
            pruned_entity, pruned_relations, node_map = pruning_agent.prune(
                entity_list, relation_list1, stage
            )
            # ──────────────────────────────────────────────────────────────

            parameter = [
                CXPB, MUTPB, NGEN, popsize,
                stage_regu_len,
                stage_index_regu_dic, stage_index_tech_dic,
                pruned_entity,      # ← đồ thị đã cắt
                pruned_relations,   # ← đồ thị đã cắt
                stage,
            ]

            # ── FastGA thay cho GA ─────────────────────────────────────────
            run = FastGA(parameter)
            (best_gene,
             best_entity_regu_dic_pruned,
             best_relation_regu_dic_pruned) = run.GA_main()
            # ──────────────────────────────────────────────────────────────

            # ── Remap về index gốc ────────────────────────────────────────
            bestindividual_entity_regu_dic   = pruning_agent.remap_entity_dic(
                best_entity_regu_dic_pruned, node_map
            )
            bestindividual_relation_regu_dic = pruning_agent.remap_relation_dic(
                best_relation_regu_dic_pruned, node_map
            )
            # ──────────────────────────────────────────────────────────────

            logger.info(f"Stage {stage} done | entity_regu_dic: {bestindividual_entity_regu_dic}")
            logger.info(f"Stage {stage} done | relation_regu_dic: {bestindividual_relation_regu_dic}")

            # ── Xử lý unsuccess_edge (giữ nguyên logic gốc) ───────────────
            unsuccess_edge = [[]]
            for key in bestindividual_relation_regu_dic.keys():
                if len(bestindividual_relation_regu_dic[key]) == 0:
                    target_relation = find_target_relation(relation_list1[stage - 1], key)
                    if target_relation:
                        unsuccess_edge[0].append(target_relation)

            if unsuccess_edge[0]:
                new_target_stage = 2 if stage == 1 else stage - 1
                stage_index_regu_dic2 = DL.load_regulation()[new_target_stage - 1]
                stage_index_tech_dic2 = DL.load_tech()[new_target_stage - 1]
                stage_regu_len2       = stage_len[new_target_stage - 1]

                # Prune cũng cho lần retry
                p_entity2, p_relations2, node_map2 = pruning_agent.prune(
                    entity_list, unsuccess_edge, 1
                )
                parameter2 = [
                    CXPB, MUTPB, NGEN, popsize,
                    stage_regu_len2,
                    stage_index_regu_dic2, stage_index_tech_dic2,
                    p_entity2, p_relations2, 1,
                ]
                run2 = FastGA(parameter2)
                (_, best_e2, best_r2) = run2.GA_main()

                best_e2 = pruning_agent.remap_entity_dic(best_e2, node_map2)
                best_r2 = pruning_agent.remap_relation_dic(best_r2, node_map2)

                for key1 in best_r2.keys():
                    if best_r2[key1]:
                        info = f"{new_target_stage}-{best_r2[key1][0]}"
                        bestindividual_relation_regu_dic[key1].append(info)
                        target_relation = find_target_relation(relation_list1[stage - 1], key1)
                        if target_relation:
                            e1 = int(target_relation[0])
                            e2 = int(target_relation[1])
                            bestindividual_entity_regu_dic.setdefault(e1, []).append(info)
                            bestindividual_entity_regu_dic.setdefault(e2, []).append(info)

            # Cập nhật entity_instance_dic / relation_instance_dic (giữ nguyên logic gốc)
            for key in list(bestindividual_entity_regu_dic.keys()):
                for rule in bestindividual_entity_regu_dic[key]:
                    info = rule if "-" in str(rule) else f"{stage}-{rule}"
                    if info not in entity_instance_dic.get(key, []):
                        entity_instance_dic.setdefault(key, []).append(info)

            for key in list(bestindividual_relation_regu_dic.keys()):
                for rule in bestindividual_relation_regu_dic[key]:
                    info = rule if "-" in str(rule) else f"{stage}-{rule}"
                    if info not in relation_instance_dic.get(key, []):
                        relation_instance_dic.setdefault(key, []).append(info)

        # ── Phần ghi file giữ nguyên hoàn toàn từ gốc ────────────────────
        entity_one_instance_dic = {}
        for key in entity_instance_dic.keys():
            entity_one_instance_dic[key] = 0
            if entity_instance_dic[key]:
                target_rule = entity_instance_dic[key][
                    random.randint(0, len(entity_instance_dic[key]) - 1)
                ]
                target_type  = entity_list[int(key)]
                target_stage = int(str(target_rule)[0])
                target_rule_index = int(str(target_rule).split("-")[1]) - 1
                target_tech_dic = DL.load_tech()[target_stage - 1]
                target_tech = target_tech_dic[list(target_tech_dic.keys())[target_rule_index]]

                target_instance_list = []
                for tech in target_tech:
                    for data in instance_lib:
                        if data["stage-key"] == f"{tech}-{os_type}":
                            for instance in data.get(target_type, []):
                                target_instance_list.append(instance)

                entity_one_instance_dic[key] = (
                    random.choice(target_instance_list) if target_instance_list else "#######"
                )

        not_satisfied_entity = []
        for i in range(len(entity_one_instance_dic)):
            if entity_one_instance_dic[list(entity_one_instance_dic.keys())[i]] != 0:
                graph_data[i + 1] = (
                    graph_data[i + 1].strip()
                    + "-"
                    + entity_one_instance_dic[list(entity_one_instance_dic.keys())[i]]
                    + "\n"
                )
            else:
                not_satisfied_entity.append(i + 1)

        temp_graph_data = [
            line for j, line in enumerate(graph_data) if j not in not_satisfied_entity
        ]

        delete_edge_list = []
        new_entity_match_dic = {}
        for e in range(int(temp_graph_data[0])):
            if e + 1 not in not_satisfied_entity:
                i1 = sum(1 for t in not_satisfied_entity if e + 1 > t)
                new_entity_match_dic[e] = e - i1

        for x in range(len(temp_graph_data)):
            if re.match(r"\d+\s\d+\s[A-Z]+", temp_graph_data[x]):
                parts = temp_graph_data[x].split(" ", 2)
                sub, obj = int(parts[0]), int(parts[1])
                verb = parts[2].strip()
                if (sub + 1 in not_satisfied_entity) or (obj + 1 in not_satisfied_entity):
                    delete_edge_list.append(x)
                else:
                    temp_graph_data[x] = (
                        f"{new_entity_match_dic[sub]} {new_entity_match_dic[obj]} {temp_graph_data[x][-5:]}"
                    )

        final_graph_data = []
        for v in range(len(temp_graph_data)):
            if v == 0:
                final_graph_data.append(str(len(new_entity_match_dic)) + "\n")
            elif v not in delete_edge_list:
                final_graph_data.append(temp_graph_data[v])

        if len(new_entity_match_dic) + 1 < len(final_graph_data):
            final_graph_data[len(new_entity_match_dic) + 1] = (
                str(int(final_graph_data[len(new_entity_match_dic) + 1]) - len(delete_edge_list))
                + "\n"
            )

        with open(new_file_path, "w", encoding="utf-8") as file:
            file.writelines(final_graph_data)

        count = sum(1 for v in entity_instance_dic.values() if len(v) != 0)
        coverage = count / max(len(entity_instance_dic), 1)
        logger.info(f"✓ {txt} done | coverage={coverage:.2%}")