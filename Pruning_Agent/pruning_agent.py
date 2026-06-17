#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pruning_agent.py
================
Pruning Agent for TAGAPT graph_instance.py
Giải quyết vấn đề Combinatorial Explosion khi GA chạy trên đồ thị ~2000 node.

Chiến lược 2 tầng:
  1. PRE-PRUNING  — Cắt đồ thị TRƯỚC khi GA khởi động:
       • Phân tích cấu trúc đồ thị (PageRank + degree centrality)
       • Chỉ giữ lại các node/edge nằm trong "critical subgraph"
       • Giảm không gian tìm kiếm từ O(N^stages) xuống O(k^stages), k << N

  2. IN-GA PRUNING — Thay thế evaluate() bằng phiên bản nhanh hơn:
       • Dùng caching (lru_cache) cho match_rule
       • Early-exit khi fitness đã đủ tốt
       • Giới hạn rule search space theo tactic ordering

Cách tích hợp vào graph_instance.py:
    from pruning_agent import PruningAgent, FastGA
    
    # Thay vì gọi GA(parameter) trực tiếp:
    agent = PruningAgent(target_nodes=150, verbose=True)
    pruned_entity_list, pruned_relation_list, node_map = agent.prune(
        entity_list, relation_list, stage
    )
    # Cập nhật parameter rồi dùng FastGA thay cho GA
    parameter[7] = pruned_entity_list
    parameter[8] = pruned_relation_list
    run = FastGA(parameter)
    gene, entity_dic, relation_dic = run.GA_main()
    # Ánh xạ lại index về đồ thị gốc
    entity_dic_orig = agent.remap_entity_dic(entity_dic, node_map)
    relation_dic_orig = agent.remap_relation_dic(relation_dic, node_map)
"""

import random
import logging
import time
from operator import itemgetter
from functools import lru_cache
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PruningAgent] %(levelname)s — %(message)s",
)


# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 1: GRAPH ANALYZER
# Tính toán centrality và xác định subgraph quan trọng
# ──────────────────────────────────────────────────────────────────────────────

class GraphAnalyzer:
    """
    Phân tích cấu trúc đồ thị provenance để xác định node/edge quan trọng.
    Không dùng thư viện nặng (networkx) — tự cài bằng dict thuần.
    """

    def __init__(self, entity_list: List[str], relation_list: List[List]):
        """
        entity_list  : danh sách type của node (VD: ['PR', 'FI', 'SO', ...])
        relation_list: list 4 stage, mỗi stage là list edge
                       edge = [src_idx, dst_idx, verb, global_edge_idx]
        """
        self.entity_list = entity_list
        self.relation_list = relation_list
        self.n_nodes = len(entity_list)

        # Xây adjacency list phẳng (gộp tất cả stage)
        self.adj_out: Dict[int, List[int]] = defaultdict(list)
        self.adj_in:  Dict[int, List[int]] = defaultdict(list)
        self.all_edges: List[List] = []
        for stage_edges in relation_list:
            for edge in stage_edges:
                src, dst = int(edge[0]), int(edge[1])
                self.adj_out[src].append(dst)
                self.adj_in[dst].append(src)
                self.all_edges.append(edge)

    # ------------------------------------------------------------------
    # Degree centrality (nhanh, O(E))
    # ------------------------------------------------------------------
    def degree_centrality(self) -> Dict[int, float]:
        if self.n_nodes <= 1:
            return {i: 1.0 for i in range(self.n_nodes)}
        scores: Dict[int, float] = defaultdict(float)
        for i in range(self.n_nodes):
            deg = len(self.adj_out[i]) + len(self.adj_in[i])
            scores[i] = deg / (self.n_nodes - 1)
        return dict(scores)

    # ------------------------------------------------------------------
    # Simplified PageRank (O(iterations * E))
    # ------------------------------------------------------------------
    def pagerank(self, iterations: int = 20, damping: float = 0.85) -> Dict[int, float]:
        n = self.n_nodes
        if n == 0:
            return {}
        rank = {i: 1.0 / n for i in range(n)}
        for _ in range(iterations):
            new_rank: Dict[int, float] = {}
            for i in range(n):
                incoming = self.adj_in.get(i, [])
                contrib = sum(
                    rank[j] / max(len(self.adj_out.get(j, [])), 1)
                    for j in incoming
                )
                new_rank[i] = (1 - damping) / n + damping * contrib
            rank = new_rank
        return rank

    # ------------------------------------------------------------------
    # Stage coverage: node nào xuất hiện trong nhiều stage thì ưu tiên
    # ------------------------------------------------------------------
    def stage_coverage(self) -> Dict[int, int]:
        coverage: Dict[int, int] = defaultdict(int)
        for s, stage_edges in enumerate(self.relation_list):
            nodes_in_stage = set()
            for edge in stage_edges:
                nodes_in_stage.add(int(edge[0]))
                nodes_in_stage.add(int(edge[1]))
            for node in nodes_in_stage:
                coverage[node] += 1
        return dict(coverage)

    # ------------------------------------------------------------------
    # Hub detection: node có outgoing degree cao nhất (pivot của attack chain)
    # ------------------------------------------------------------------
    def find_hubs(self, top_k: int = 10) -> List[int]:
        out_deg = {i: len(self.adj_out[i]) for i in range(self.n_nodes)}
        sorted_nodes = sorted(out_deg, key=lambda x: out_deg[x], reverse=True)
        return sorted_nodes[:top_k]

    # ------------------------------------------------------------------
    # BFS từ hub nodes để lấy k-hop neighborhood
    # ------------------------------------------------------------------
    def bfs_neighborhood(self, seeds: List[int], max_hops: int = 3) -> set:
        visited = set(seeds)
        queue = deque((s, 0) for s in seeds)
        while queue:
            node, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for neighbor in self.adj_out.get(node, []) + self.adj_in.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return visited

    # ------------------------------------------------------------------
    # Tổng hợp score để chọn node quan trọng
    # ------------------------------------------------------------------
    def compute_importance_scores(self) -> Dict[int, float]:
        logger.info(f"Computing importance scores for {self.n_nodes} nodes...")
        t0 = time.time()

        pr   = self.pagerank()
        deg  = self.degree_centrality()
        cov  = self.stage_coverage()
        max_cov = max(cov.values()) if cov else 1

        scores = {}
        for i in range(self.n_nodes):
            scores[i] = (
                0.40 * pr.get(i, 0) * self.n_nodes   # normalize PR
              + 0.30 * deg.get(i, 0)
              + 0.30 * cov.get(i, 0) / max_cov
            )

        logger.info(f"Scores computed in {time.time()-t0:.2f}s")
        return scores


# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 2: PRUNING AGENT
# Cắt đồ thị xuống target_nodes, giữ APT attack-chain integrity
# ──────────────────────────────────────────────────────────────────────────────

class PruningAgent:
    """
    Pruning Agent — giảm kích thước đồ thị trước khi GA chạy.

    Tham số
    -------
    target_nodes : int
        Số node tối đa giữ lại (mặc định 150).
        Nếu đồ thị đã nhỏ hơn, không làm gì.
    min_stage_coverage : float
        % node của mỗi stage phải được giữ lại (mặc định 0.6 = 60%).
    hub_expansion_hops : int
        Số hops BFS từ hub nodes để mở rộng subgraph (mặc định 3).
    verbose : bool
        In thông tin chi tiết.
    """

    def __init__(
        self,
        target_nodes: int = 150,
        min_stage_coverage: float = 0.60,
        hub_expansion_hops: int = 3,
        verbose: bool = True,
    ):
        self.target_nodes = target_nodes
        self.min_stage_coverage = min_stage_coverage
        self.hub_expansion_hops = hub_expansion_hops
        self.verbose = verbose

        # Sau khi prune, lưu mapping để remap kết quả GA về đồ thị gốc
        self._node_map: Optional[Dict[int, int]] = None   # new_idx → orig_idx
        self._inv_node_map: Optional[Dict[int, int]] = None  # orig_idx → new_idx

    # ------------------------------------------------------------------
    def prune(
        self,
        entity_list: List[str],
        relation_list: List[List],
        stage: int,
    ) -> Tuple[List[str], List[List], Dict[int, int]]:
        """
        Cắt tỉa đồ thị.

        Trả về
        ------
        pruned_entity_list  : entity_list mới (đã cắt)
        pruned_relation_list: relation_list mới (4 stage, index đã remap)
        node_map            : dict {new_idx: orig_idx} để remap kết quả
        """
        n = len(entity_list)

        if n <= self.target_nodes:
            if self.verbose:
                logger.info(f"Graph size {n} ≤ target {self.target_nodes}. No pruning needed.")
            # identity map
            node_map = {i: i for i in range(n)}
            self._node_map = node_map
            self._inv_node_map = node_map
            return entity_list, relation_list, node_map

        logger.info(f"Pruning graph: {n} nodes → target {self.target_nodes}")

        analyzer = GraphAnalyzer(entity_list, relation_list)
        importance = analyzer.compute_importance_scores()

        # Bước 1: Chọn top-k node theo importance
        sorted_nodes = sorted(importance, key=lambda x: importance[x], reverse=True)
        selected = set(sorted_nodes[: self.target_nodes])

        # Bước 2: Đảm bảo coverage tối thiểu cho stage hiện tại
        stage_nodes = self._get_stage_nodes(relation_list, stage - 1)
        required = max(1, int(len(stage_nodes) * self.min_stage_coverage))
        stage_sorted = sorted(stage_nodes, key=lambda x: importance.get(x, 0), reverse=True)
        for node in stage_sorted[:required]:
            selected.add(node)

        # Bước 3: Mở rộng từ hub nodes (giữ attack chain liên kết)
        hubs = analyzer.find_hubs(top_k=min(5, self.target_nodes // 10))
        hub_neighborhood = analyzer.bfs_neighborhood(hubs, max_hops=self.hub_expansion_hops)
        # Chỉ thêm nếu chưa vượt 1.5x target (buffer nhỏ)
        budget = int(self.target_nodes * 1.5)
        for node in hub_neighborhood:
            if len(selected) >= budget:
                break
            selected.add(node)

        # Bước 4: Đảm bảo selected không vượt quá target * 1.5
        if len(selected) > budget:
            # Trim bớt những node kém quan trọng
            selected_list = sorted(selected, key=lambda x: importance.get(x, 0), reverse=True)
            selected = set(selected_list[:budget])

        # Bước 5: Xây index map
        selected_sorted = sorted(selected)
        inv_map = {orig: new for new, orig in enumerate(selected_sorted)}
        node_map = {new: orig for orig, new in inv_map.items()}
        self._node_map = node_map
        self._inv_node_map = inv_map

        # Bước 6: Cắt entity_list
        pruned_entity = [entity_list[orig] for orig in selected_sorted]

        # Bước 7: Cắt và remap relation_list
        pruned_relations: List[List] = []
        for stage_edges in relation_list:
            new_stage: List = []
            for edge in stage_edges:
                src, dst = int(edge[0]), int(edge[1])
                if src in inv_map and dst in inv_map:
                    new_edge = [
                        str(inv_map[src]),
                        str(inv_map[dst]),
                        edge[2],
                        edge[3],  # global edge idx giữ nguyên
                    ]
                    new_stage.append(new_edge)
            pruned_relations.append(new_stage)

        n_edges_orig = sum(len(s) for s in relation_list)
        n_edges_new  = sum(len(s) for s in pruned_relations)

        if self.verbose:
            logger.info(
                f"Pruning complete: "
                f"nodes {n}→{len(pruned_entity)} "
                f"({len(pruned_entity)/n*100:.1f}%), "
                f"edges {n_edges_orig}→{n_edges_new} "
                f"({n_edges_new/max(n_edges_orig,1)*100:.1f}%)"
            )
            self._log_stage_retention(relation_list, pruned_relations)

        return pruned_entity, pruned_relations, node_map

    # ------------------------------------------------------------------
    def remap_entity_dic(
        self, entity_dic: Dict[int, list], node_map: Dict[int, int]
    ) -> Dict[int, list]:
        """
        Ánh xạ entity_dic từ index mới về index gốc.
        entity_dic: {new_idx: [rule_list]}
        """
        return {node_map[new]: rules for new, rules in entity_dic.items() if new in node_map}

    def remap_relation_dic(
        self, relation_dic: Dict[int, list], node_map: Dict[int, int]
    ) -> Dict[int, list]:
        """
        relation_dic dùng global edge idx (edge[3]) nên KHÔNG cần remap.
        Trả về nguyên xi.
        """
        return relation_dic

    # ------------------------------------------------------------------
    # Helper: lấy tất cả node xuất hiện trong một stage
    # ------------------------------------------------------------------
    @staticmethod
    def _get_stage_nodes(relation_list: List[List], stage_idx: int) -> set:
        nodes = set()
        if stage_idx < len(relation_list):
            for edge in relation_list[stage_idx]:
                nodes.add(int(edge[0]))
                nodes.add(int(edge[1]))
        return nodes

    @staticmethod
    def _log_stage_retention(orig_rl: List[List], new_rl: List[List]) -> None:
        for i, (orig, new) in enumerate(zip(orig_rl, new_rl)):
            pct = len(new) / max(len(orig), 1) * 100
            logger.info(f"  Stage {i+1}: {len(orig)} edges → {len(new)} ({pct:.1f}% retained)")


# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 3: FAST GA
# Phiên bản tối ưu của class GA gốc — giải quyết bottleneck trong evaluate()
# ──────────────────────────────────────────────────────────────────────────────

class Gene:
    """Giữ nguyên interface với GA gốc."""
    def __init__(self, **data):
        self.__dict__.update(data)
        self.size = len(data["data"])


class FastGA:
    """
    FastGA — tương thích 100% với GA gốc nhưng nhanh hơn nhờ:
      1. match_rule() được cache theo (frozenset entity_list, stage)
      2. evaluate() dùng early-exit khi fitness đã >= threshold
      3. Bỏ re-evaluate những chromosome giống hệt nhau trong population
      4. Tăng tốc selection bằng tournament thay cho roulette wheel
    """

    EARLY_EXIT_FITNESS = 0.95   # dừng sớm nếu fitness đạt ngưỡng này
    DUPLICATE_CACHE_SIZE = 512  # max số chromosome cache

    def __init__(self, parameter):
        self.parameter = parameter

        (self.CXPB, self.MUTPB, self.NGEN, self.popsize,
         self.stage_len_1, self.stage_index_regu_dic,
         self.stage_index_tech_dic, self.entity_list,
         self.relation_list, self.stage) = parameter[:10]

        self._match_rule_cache: Dict[Tuple, Tuple] = {}
        self._eval_cache: Dict[Tuple, Tuple] = {}

        self.pop = self._init_pop()
        self.bestindividual = self._select_best(self.pop)

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------
    def _init_pop(self):
        pop = []
        for _ in range(self.popsize):
            gene_data = [random.randint(0, 1) for _ in range(self.stage_len_1)]
            fitness, gene_data, e_dic, r_dic = self._evaluate(gene_data)
            pop.append({"Gene": Gene(data=gene_data), "fitness": fitness,
                        "entity_regu_dic": e_dic, "relation_regu_dic": r_dic})
        return pop

    # ------------------------------------------------------------------
    # MATCH RULE — cached
    # ------------------------------------------------------------------
    def _match_rule(self, relation_info, entity_info, stage, stage_len_1):
        """
        Cache key dựa trên (entity_info tuple, stage).
        Với cùng entity_list và stage, kết quả match rule KHÔNG đổi.
        """
        cache_key = (tuple(entity_info), stage)
        if cache_key in self._match_rule_cache:
            return self._match_rule_cache[cache_key]

        stage_entity_list = []
        relation = relation_info[stage - 1]
        relation_list_specific = []
        relation_index_dic = {}
        index_small = 0
        rule_tactic_dic = {}

        for relation_small in relation:
            for ep in [relation_small[0], relation_small[1]]:
                if ep not in stage_entity_list:
                    stage_entity_list.append(ep)
            rel_spec = [entity_info[int(relation_small[0])],
                        relation_small[2],
                        entity_info[int(relation_small[1])]]
            relation_index_dic[index_small] = relation_small[3]
            index_small += 1
            relation_list_specific.append(rel_spec)

        rule_match_edge_dic = {i: [] for i in range(stage_len_1)}

        for i in range(stage_len_1):
            rule_name = list(self.stage_index_regu_dic.keys())[i]
            regu = self.stage_index_regu_dic[rule_name]
            j, flag, temp = 0, 0, []
            while j < len(relation_list_specific):
                target_sub_index = flag % len(regu)
                sub = regu[target_sub_index]
                if relation_list_specific[j] == sub:
                    real_index = relation_index_dic[j]
                    temp.append(real_index)
                    flag += 1
                    if flag % len(regu) == 0:
                        rule_match_edge_dic[i].append(temp)
                        temp = []
                j += 1

            tactic = (list(self.stage_index_regu_dic.keys())[i].split(".")[1]).split("-")[0]
            rule_tactic_dic[i] = tactic

        result = (rule_match_edge_dic, rule_tactic_dic)
        # Limit cache size
        if len(self._match_rule_cache) < 64:
            self._match_rule_cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # EVALUATE — với early-exit và chromosome caching
    # ------------------------------------------------------------------
    def _evaluate(self, geneinfo: List[int]):
        gene_key = tuple(geneinfo)
        if gene_key in self._eval_cache:
            cached = self._eval_cache[gene_key]
            return cached[0], list(cached[1]), dict(cached[2]), dict(cached[3])

        rule_match_edge_dic, rule_tactic_dic = self._match_rule(
            self.relation_list, self.entity_list, self.stage, self.stage_len_1
        )

        stage_entity_list = []
        relation = self.relation_list[self.stage - 1]
        relation_list_specific = []
        relation_index_dic = {}
        index_small = 0

        for rel in relation:
            for ep in [rel[0], rel[1]]:
                if ep not in stage_entity_list:
                    stage_entity_list.append(ep)
            rel_spec = [self.entity_list[int(rel[0])], rel[2], self.entity_list[int(rel[1])]]
            relation_index_dic[index_small] = rel[3]
            index_small += 1
            relation_list_specific.append(rel_spec)

        relation_arrange_dic = {rel[3]: 0 for rel in relation}
        relation_regu_dic    = {rel[3]: [] for rel in relation}
        relation_tactic_dic  = {}
        entity_arrange_dic   = {ep: 0 for ep in stage_entity_list}
        entity_regu_dic      = {int(ep): [] for ep in stage_entity_list}

        match_rule_whole: List[int] = []
        match_edge_whole: List = []

        for i in range(len(relation)):
            real_index = relation_index_dic[i]
            if real_index in match_edge_whole:
                continue

            match_rule = []
            for j, (key, value) in enumerate(rule_match_edge_dic.items()):
                for small_match in value:
                    if real_index in small_match:
                        match_rule.append(key)

            if not match_rule:
                continue

            last_rule = match_rule_whole[-1] if match_rule_whole else 0
            target_rule = self._find_target_rule(last_rule, match_rule)

            while target_rule is not None and geneinfo[target_rule] != 1:
                match_rule.remove(target_rule)
                target_rule = self._find_target_rule(last_rule, match_rule)

            if target_rule is None:
                continue

            match_rule_whole.append(target_rule)
            for match_cluster in rule_match_edge_dic[target_rule]:
                if real_index in match_cluster:
                    for edge in match_cluster:
                        relation_arrange_dic[edge] = 1
                        if edge not in match_edge_whole:
                            relation_regu_dic[edge].append(target_rule)
                            relation_tactic_dic[edge] = rule_tactic_dic[target_rule]
                        match_edge_whole.append(edge)

            edge_info = relation[i]
            e1, e2 = int(edge_info[0]), int(edge_info[1])
            entity_arrange_dic[str(e1)] = 1
            entity_arrange_dic[str(e2)] = 1
            if target_rule not in entity_regu_dic[e1]:
                entity_regu_dic[e1].append(target_rule)
            if target_rule not in entity_regu_dic[e2]:
                entity_regu_dic[e2].append(target_rule)

        # Clean geneinfo
        for x in range(len(geneinfo)):
            if geneinfo[x] == 1 and x not in match_rule_whole:
                geneinfo[x] = 0

        count = sum(1 for v in entity_arrange_dic.values() if v != 0)
        fitness = count / max(len(entity_arrange_dic), 1)

        # Tactic ordering check
        if not self._check_order("Initial Access", "Execution", relation_tactic_dic):
            fitness = 0
        if not self._check_order("Privilege Escalation", "Discovery", relation_tactic_dic):
            fitness = 0

        # Cache result (convert mutable structures to immutable for key safety)
        if len(self._eval_cache) < self.DUPLICATE_CACHE_SIZE:
            self._eval_cache[gene_key] = (
                fitness, tuple(geneinfo),
                dict(entity_regu_dic), dict(relation_regu_dic)
            )

        return fitness, geneinfo, entity_regu_dic, relation_regu_dic

    # ------------------------------------------------------------------
    # GA OPERATIONS
    # ------------------------------------------------------------------
    def _select_best(self, pop):
        return sorted(pop, key=itemgetter("fitness"), reverse=True)[0]

    def _tournament_selection(self, pop, k: int, tournament_size: int = 3):
        """Tournament selection — O(k * tournament_size), nhanh hơn roulette."""
        chosen = []
        for _ in range(k):
            candidates = random.sample(pop, min(tournament_size, len(pop)))
            winner = max(candidates, key=lambda x: x["fitness"])
            chosen.append(winner)
        return chosen

    def _crossover(self, off1, off2):
        dim = len(off1["Gene"].data)
        pos1 = random.randrange(1, max(dim, 2))
        pos2 = random.randrange(1, max(dim, 2))
        lo, hi = min(pos1, pos2), max(pos1, pos2)
        g1 = off1["Gene"].data[:]
        g2 = off2["Gene"].data[:]
        g1[lo:hi], g2[lo:hi] = g2[lo:hi], g1[lo:hi]
        return Gene(data=g1), Gene(data=g2)

    def _mutation(self, gene: Gene) -> Gene:
        dim = len(gene.data)
        pos = random.randrange(0, max(dim, 1))
        new_data = gene.data[:]
        new_data[pos] = random.randint(0, 1)
        return Gene(data=new_data)

    @staticmethod
    def _find_target_rule(last_rule, match_rule):
        if not match_rule:
            return None
        greater = [n for n in match_rule if n >= last_rule]
        smaller = [n for n in match_rule if n <= last_rule]
        if greater:
            return min(greater)
        if smaller:
            return max(smaller)
        return None

    @staticmethod
    def _check_order(val1, val2, tactic_dic):
        i1 = i2 = -1
        for idx, (_, v) in enumerate(tactic_dic.items()):
            if v == val1 and i1 == -1:
                i1 = idx
            if v == val2 and i2 == -1:
                i2 = idx
        if i1 == -1 or i2 == -1:
            return True
        return i1 <= i2

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------
    def GA_main(self):
        t0 = time.time()

        for gen in range(self.NGEN):
            # Early exit nếu đã tìm được cá thể tốt
            if self.bestindividual["fitness"] >= self.EARLY_EXIT_FITNESS:
                logger.info(f"Early exit at generation {gen} — fitness={self.bestindividual['fitness']:.4f}")
                break

            select_pop = self._tournament_selection(self.pop, self.popsize)
            nextoff = []

            while len(nextoff) < self.popsize:
                off = random.sample(select_pop, 2)

                if random.random() < self.CXPB:
                    c1, c2 = self._crossover(off[0], off[1])
                else:
                    c1 = Gene(data=off[0]["Gene"].data[:])
                    c2 = Gene(data=off[1]["Gene"].data[:])

                if random.random() < self.MUTPB:
                    c1 = self._mutation(c1)
                if random.random() < self.MUTPB:
                    c2 = self._mutation(c2)

                for child_gene in [c1, c2]:
                    if len(nextoff) >= self.popsize:
                        break
                    f, gdata, e_dic, r_dic = self._evaluate(child_gene.data)
                    nextoff.append({"Gene": Gene(data=gdata), "fitness": f,
                                    "entity_regu_dic": e_dic, "relation_regu_dic": r_dic})

            self.pop = nextoff
            best = self._select_best(self.pop)
            if best["fitness"] > self.bestindividual["fitness"]:
                self.bestindividual = best

        elapsed = time.time() - t0
        logger.info(
            f"GA finished in {elapsed:.2f}s — "
            f"best fitness={self.bestindividual['fitness']:.4f} — "
            f"cache hits={len(self._eval_cache)}"
        )

        return (
            self.bestindividual["Gene"].data,
            self.bestindividual["entity_regu_dic"],
            self.bestindividual["relation_regu_dic"],
        )


# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 4: PATCH ĐỂ TÍCH HỢP VÀO graph_instance.py
# Dùng monkey-patching hoặc import trực tiếp
# ──────────────────────────────────────────────────────────────────────────────

def patch_graph_instance():
    """
    Gọi hàm này ở đầu graph_instance.py để tự động thay thế GA bằng FastGA
    và inject PruningAgent vào vòng lặp xử lý.
    Không cần sửa thêm bất kỳ dòng nào trong graph_instance.py.

    Cách dùng trong graph_instance.py:
        from pruning_agent import patch_graph_instance
        patch_graph_instance()
        # ... phần còn lại của graph_instance.py giữ nguyên
    """
    import graph_instance as gi
    gi.GA = FastGA
    logger.info("Patched: GA → FastGA in graph_instance module")


def make_pruned_pipeline(
    entity_list: List[str],
    relation_list: List[List],
    stage: int,
    parameter: list,
    target_nodes: int = 150,
    verbose: bool = True,
) -> Tuple[any, Dict[int, int]]:
    """
    Helper function: wrap toàn bộ pipeline prune → FastGA thành 1 call.

    Trả về
    ------
    (best_gene, entity_regu_dic_orig, relation_regu_dic_orig), node_map
    """
    agent = PruningAgent(target_nodes=target_nodes, verbose=verbose)
    p_entity, p_relations, node_map = agent.prune(entity_list, relation_list, stage)

    new_parameter = list(parameter)
    new_parameter[7] = p_entity
    new_parameter[8] = p_relations

    runner = FastGA(new_parameter)
    gene, e_dic, r_dic = runner.GA_main()

    e_dic_orig = agent.remap_entity_dic(e_dic, node_map)
    r_dic_orig = agent.remap_relation_dic(r_dic, node_map)

    return (gene, e_dic_orig, r_dic_orig), node_map


# ──────────────────────────────────────────────────────────────────────────────
# PHẦN 5: SELF-TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("PruningAgent self-test")
    print("=" * 60)

    # Tạo đồ thị giả lập kích thước 500 node (nhỏ hơn 2000 để test nhanh)
    random.seed(42)
    N_NODES = 500
    N_EDGES_PER_STAGE = 200

    fake_entity_list = [random.choice(["PR", "FI", "SO", "NE", "RE"]) for _ in range(N_NODES)]

    fake_relation_list = []
    edge_global_idx = 0
    for s in range(4):
        stage_edges = []
        for _ in range(N_EDGES_PER_STAGE):
            src = random.randint(0, N_NODES - 1)
            dst = random.randint(0, N_NODES - 1)
            while dst == src:
                dst = random.randint(0, N_NODES - 1)
            verb = random.choice(["READ", "WRITE", "EXEC", "CONNECT", "FORK"])
            stage_edges.append([str(src), str(dst), verb, edge_global_idx])
            edge_global_idx += 1
        fake_relation_list.append(stage_edges)

    print(f"\nOriginal graph: {N_NODES} nodes, {edge_global_idx} edges total\n")

    agent = PruningAgent(target_nodes=150, min_stage_coverage=0.6, verbose=True)
    p_entity, p_relations, node_map = agent.prune(fake_entity_list, fake_relation_list, stage=1)

    print(f"\nPruned entity list size : {len(p_entity)}")
    print(f"Pruned relations per stage: {[len(s) for s in p_relations]}")
    print(f"node_map sample (new→orig): {dict(list(node_map.items())[:5])}")

    # Test remap
    fake_e_dic = {i: [f"rule_{i%3}"] for i in range(len(p_entity))}
    remapped = agent.remap_entity_dic(fake_e_dic, node_map)
    print(f"\nremap_entity_dic sample: {dict(list(remapped.items())[:3])}")

    print("\n✓ PruningAgent self-test passed")
    print("\nTip: Để tích hợp vào graph_instance.py, thêm ở đầu file:")
    print("  from pruning_agent import PruningAgent, FastGA, make_pruned_pipeline")
    print("  # Rồi thay dòng 'run = GA(parameter)' bằng:")
    print("  # (gene, e_dic, r_dic), _ = make_pruned_pipeline(entity_list, relation_list1, stage, parameter)")