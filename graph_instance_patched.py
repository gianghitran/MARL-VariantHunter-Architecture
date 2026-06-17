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
  5. ADDED: Sinh ra file visualize đồ thị (Graph Visualization) sau mỗi file thành công!
"""

import os
import random
import re
import json
import logging
import networkx as nx
import matplotlib.pyplot as plt
from pruning_agent import PruningAgent, FastGA
from edge_validator import EdgeConstraintValidator

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
            line = line.strip()
            # Match edge lines: "0 1 FR" or "0 1 FR-1" or "0 1 FR-1-2-3"
            # Stage annotation is OPTIONAL — defaults to all 4 stages if missing
            m = re.match(r'^(\d+)\s+(\d+)\s+([A-Z]{2})(?:(-[\d-]+))?$', line)
            if not m:
                continue
            num1, num2, verb, stage_str = m.groups()
            relation = [num1, num2, verb, i1]
            i1 += 1
            if stage_str:
                # Parse stage annotations like "-1", "-1-2", "-1-2-3"
                stages = [int(s) for s in stage_str.split('-') if s.strip()]
                for stage_num in stages:
                    if 1 <= stage_num <= 4:
                        relation_list[stage_num - 1].append(list(relation))
            else:
                # No stage annotation — assign to ALL stages
                for s in range(4):
                    relation_list[s].append(list(relation))
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

def visualize_graph_from_data(graph_data, save_path):
    """
    Vẽ đồ thị và lưu thành ảnh PNG từ dữ liệu văn bản của đồ thị provenance
    """
    try:
        num_nodes = int(graph_data[0].strip())
        nodes = [line.strip() for line in graph_data[1:num_nodes+1]]
        
        num_edges = int(graph_data[num_nodes+1].strip())
        edges = [line.strip() for line in graph_data[num_nodes+2:num_nodes+2+num_edges]]
        
        G = nx.DiGraph()
        
        # Thêm nodes
        for i, node_label in enumerate(nodes):
            # Cắt ngắn label nếu quá dài để hiển thị đồ thị đẹp hơn
            short_label = node_label.split('-')[0] + '-' + node_label.split('-')[1][:15] if '-' in node_label else node_label
            G.add_node(i, label=short_label)
            
        # Thêm edges
        for edge in edges:
            parts = edge.split()
            if len(parts) >= 3:
                u, v, action = int(parts[0]), int(parts[1]), parts[2]
                G.add_edge(u, v, label=action)
                
        plt.figure(figsize=(12, 10))
        # Sử dụng thuật toán kkamada_kawai hoặc spring để layout đẹp
        pos = nx.spring_layout(G, k=1.0)
        
        nx.draw_networkx_nodes(G, pos, node_size=600, node_color='skyblue', edgecolors='black')
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=15, alpha=0.8)
        
        node_labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_family='sans-serif')
        
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, bbox=dict(alpha=0))
        
        plt.title(f"Provenance Graph Visualization\nNodes: {num_nodes} | Edges: {num_edges}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path.replace('png', 'pdf'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error(f"Lỗi khi visualize graph {save_path}: {e}")


if __name__ == "__main__":

    DL = Dataloader()
    # (Tạm comment vì file gốc chưa có sẵn data dictionary, giả định user đã có file)
    try:
        (stage1_index_regu_dic, stage2_index_regu_dic,
         stage3_index_regu_dic, stage4_index_regu_dic) = DL.load_regulation()
        (stage1_index_tech_dic, stage2_index_tech_dic,
         stage3_index_tech_dic, stage4_index_tech_dic) = DL.load_tech()
    except Exception as e: # Nếu lỗi thì bypass chạy thử
        pass

    instance_lib = read_new(r".\instance_lib\technique-instance-lib-os-filter-add.json")
    
    new_instance_path = r".\4000_3_generated_data_new2_sub_instance_windows"
    make_directory(new_instance_path)
    
    # ── Folder visualize ────────────────────────────────────────────────────────
    visualize_folder = r".\graph_visualize"
    make_directory(visualize_folder)
    # ─────────────────────────────────────────────────────────────────────────────
    
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

        # Debug: show what was parsed
        edge_counts = [len(s) for s in relation_list1]
        total_edges = sum(edge_counts)
        logger.info(f"  [DEBUG] Parsed: {len(entity_list)} entities, {total_edges} edges {edge_counts}")

        if total_edges == 0:
            logger.info(f"  [SKIP] {txt}: no edges parsed, skipping")
            continue

        # -- PRE-GA: Edge Constraint Validation --
        validator = EdgeConstraintValidator(verbose=True)
        relation_list1, filter_stats = validator.filter_relation_list(
            entity_list, relation_list1
        )

        # Debug: show what survived
        post_counts = [len(s) for s in relation_list1]
        logger.info(f"  [DEBUG] After filter: {sum(post_counts)} edges {post_counts} "
                    f"(removed {filter_stats['removed']})")

        if not filter_stats["graph_valid"]:
            logger.info(f"  [SKIP] {txt}: {filter_stats['removal_rate']:.0%} edges invalid, skipping graph")
            continue
        if filter_stats["removed"] > 0:
            # Clean up dangling nodes after edge removal
            entity_list, relation_list1, dangle_map = validator.remove_dangling_nodes(
                entity_list, relation_list1
            )
            # Update graph_data line count if nodes were removed
            if len(entity_list) < int(graph_data[0]):
                graph_data[0] = str(len(entity_list)) + "\n"

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

        # --- NEW CODE: Build out_verbs and in_verbs for validation ---
        out_verbs = {}
        in_verbs = {}
        for x in range(1, len(graph_data)):
            if re.match(r"\d+\s\d+\s[A-Z]+", graph_data[x]):
                parts = graph_data[x].strip().split()
                if len(parts) >= 3:
                    sub = parts[0]
                    obj = parts[1]
                    verb = parts[2].split('-')[0]
                    if sub not in out_verbs: out_verbs[sub] = []
                    if obj not in in_verbs: in_verbs[obj] = []
                    out_verbs[sub].append(verb)
                    in_verbs[obj].append(verb)
        # -------------------------------------------------------------

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

                # --- NEW CODE: Filter instances ---
                filtered_instances = []
                for instance in target_instance_list:
                    valid = True
                    for verb in out_verbs.get(str(key), []):
                        if not validator.validate_instance(instance, verb, "unknown", src_type=target_type):
                            valid = False
                            break
                    if not valid: continue
                    for verb in in_verbs.get(str(key), []):
                        if not validator.validate_instance("unknown", verb, instance):
                            valid = False
                            break
                    if valid:
                        filtered_instances.append(instance)
                # ----------------------------------

                if len(filtered_instances) != 0:
                    entity_one_instance_dic[key] = random.choice(filtered_instances)
                else:
                    # NO FALLTHROUGH to unvalidated pool.
                    # If no CTI instance passes the allowlist, use fallback.
                    entity_one_instance_dic[key] = validator.get_fallback_instance(target_type)

        # ── POST-ASSIGNMENT ENFORCEMENT ──
        # Catch cases where the initial per-verb filter missed cross-edge conflicts
        for node_key in list(entity_one_instance_dic.keys()):
            instance_name = entity_one_instance_dic[node_key]
            if instance_name == 0:
                continue
            node_type = entity_list[int(node_key)]
            is_valid = True
            for v in out_verbs.get(str(node_key), []):
                if not validator.validate_instance(instance_name, v, "unknown", src_type=node_type):
                    is_valid = False
                    break
            if is_valid:
                for v in in_verbs.get(str(node_key), []):
                    if not validator.validate_instance("unknown", v, instance_name):
                        is_valid = False
                        break
            if not is_valid:
                fallback = validator.get_fallback_instance(node_type)
                logger.info(f"  [ENFORCE] Node {node_key}({node_type}): '{instance_name}' failed edge validation -> fallback '{fallback}'")
                entity_one_instance_dic[node_key] = fallback

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

        # ── FINAL EDGE-PRUNING PASS ──
        instance_lookup = {}
        for key, val in entity_one_instance_dic.items():
            if val != 0:
                instance_lookup[int(key)] = val
        type_lookup = {}
        for key in entity_one_instance_dic.keys():
            idx = int(key)
            if idx < len(entity_list):
                type_lookup[idx] = entity_list[idx]

        pruned_count = 0
        for x in range(len(temp_graph_data)):
            if x in delete_edge_list:
                continue
            if re.match(r"\d+\s\d+\s[A-Z]+", temp_graph_data[x]):
                parts = temp_graph_data[x].strip().split()
                sub_idx = int(parts[0])
                obj_idx = int(parts[1])
                verb = parts[2].split('-')[0]

                new_to_old = {v: k for k, v in new_entity_match_dic.items()}
                orig_sub = new_to_old.get(sub_idx, sub_idx)
                orig_obj = new_to_old.get(obj_idx, obj_idx)

                sub_instance = instance_lookup.get(orig_sub, "")
                obj_instance = instance_lookup.get(orig_obj, "")
                sub_type = type_lookup.get(orig_sub, "TP")

                if sub_instance and obj_instance:
                    if not validator.validate_instance(
                        sub_instance, verb, obj_instance, src_type=sub_type
                    ):
                        delete_edge_list.append(x)
                        pruned_count += 1

        if pruned_count > 0:
            logger.info(f"  [PRUNE] Removed {pruned_count} edges that violated "
                        f"instance-level allowlist after CTI assignment")

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
            
        # =========================================================
        # VISUALIZE VÀ LƯU ẢNH CUỐI VÒNG LẶP CHO MỖI QUÁ TRÌNH SINH VÀ ĐỌC FILE
        # =========================================================
        base_name = os.path.splitext(txt)[0]
        vis_save_path = os.path.join(visualize_folder, f"{base_name}_viz.png")
        visualize_graph_from_data(final_graph_data, vis_save_path)
        logger.info(f"✓ Saved visualization graph to: {vis_save_path}")
        # =========================================================

        count = sum(1 for v in entity_instance_dic.values() if len(v) != 0)
        coverage = count / max(len(entity_instance_dic), 1)
        logger.info(f"✓ {txt} done | coverage={coverage:.2%}")
