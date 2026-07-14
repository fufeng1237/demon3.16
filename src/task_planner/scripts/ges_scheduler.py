#!/usr/bin/env python3
"""
GES — Guided Ejection Search (引导弹射搜索, 精简版)
核心: 弹射链 (ejection chain) + 引导扰动 + 贪心修复

论文出处:
  Curtois, T., Landa-Silva, D., Petrovic, S., Qu, Y. (2018).
  "Large Neighbourhood Search with Adaptive Guided Ejection Search for the PDPTW."
  EURO Journal on Transportation and Logistics, 7(2), 151–192.

  NVIDIA cuOpt (Çördük, Sielski & Chung, 2024) 用 GPU 并行 GES 在
  Li & Lim PDPTW 基准上打破 8 项世界纪录。

算法原理:
  1. 当插入请求导致不可行时，不直接拒绝，而是"弹射"出冲突请求
  2. 弹射的请求被放入待重新插入的池子
  3. 对池中每个请求按"易插入度"排序（guided 的关键）
  4. 将所有弹射出的请求重新贪心插入
  5. 每一轮迭代随机选一个任务，尝试 relocation + ejection chain

区分特征:
  - 与 ALNS 的差异: GES 不做大规模 destroy/repair，而是精确弹射冲突任务
  - 与 Tabu 的差异: GES 通过弹射链 (而非禁忌表) 跳出局部最优
  - 与 VNS 的差异: GES 只有一个核心操作（弹射 relocate），不切换邻域
"""
import random
from copy import deepcopy
from typing import Dict, List
from base_scheduler import BaseScheduler, RouteNode


class GESScheduler(BaseScheduler):

    def __init__(self, road_network, ships, tasks):
        super().__init__(road_network, ships, tasks)
        self.max_iter = 300
        self.max_ejection_depth = 3  # 最大弹射深度
        self.tabu_tenure = 10  # 简单禁忌防震荡

    def optimize(self) -> Dict[int, List[RouteNode]]:
        routes = self.greedy_init()
        best_routes = deepcopy(routes)
        best_ms = self.makespan(best_routes)

        current_routes = deepcopy(routes)
        tabu = {}  # (tid, old_ship) → remaining_tenure

        for it in range(self.max_iter):
            tasks = self._collect_all_tasks(current_routes)
            if len(tasks) < 1:
                continue

            # 随机选一个任务 + 目标船
            sid_from, tid = random.choice(tasks)
            other_ships = [s for s in self.ships if s != sid_from]
            if not other_ships:
                continue
            sid_to = random.choice(other_ships)

            # Tabu 检查
            move_key = (tid, sid_from)
            if tabu.get(move_key, 0) > 0:
                continue

            # ── 尝试直接 relocation ──
            new_routes = self._copy_routes(current_routes)
            self._remove_task(new_routes, sid_from, tid)
            task = self.tasks.get(tid)
            if task is None:
                continue

            pu, de, ok = self._find_insertion(self.ships[sid_to], new_routes[sid_to], task)

            if ok:
                # 直接可行
                self._do_insert(new_routes, sid_to, tid, pu, de, task)
                ms = self.makespan(new_routes)

                if ms < best_ms:
                    best_routes = deepcopy(new_routes)
                    best_ms = ms
                    if best_key := move_key:
                        tabu[best_key] = self.tabu_tenure
                    current_routes = new_routes
                elif ms < best_ms * 1.02:
                    # 小幅退化接受 (类似 SA)
                    current_routes = new_routes
                    if best_key := move_key:
                        tabu[best_key] = self.tabu_tenure
            else:
                # ── 弹射链: 目标船不可行，弹射冲突任务 ──
                result = self._ejection_relocate(current_routes, sid_from, sid_to, tid,
                                                 depth=self.max_ejection_depth)
                if result is not None:
                    new_routes, ejected_count = result
                    ms = self.makespan(new_routes)
                    if ms < best_ms:
                        best_routes = deepcopy(new_routes)
                        best_ms = ms
                        current_routes = new_routes
                        if best_key := move_key:
                            tabu[best_key] = self.tabu_tenure
                    elif ms < best_ms * 1.02 and ejected_count <= 2:
                        current_routes = new_routes
                        if best_key := move_key:
                            tabu[best_key] = self.tabu_tenure

            # 衰减禁忌
            expired = [k for k, v in tabu.items() if v <= 0]
            for k in expired:
                del tabu[k]
            for k in list(tabu.keys()):
                tabu[k] -= 1

        # ── 最终微调: 尝试降低 makespan ──
        best_routes = self._post_optimize(best_routes)

        return best_routes

    def _ejection_relocate(self, routes, sid_from, sid_to, tid, depth=3):
        """
        弹射链 relocate:
          1. 将 tid 从 sid_from 移除
          2. 尝试插入 sid_to, 找出冲突任务
          3. 弹射冲突任务 + 额外扰动
          4. 重新贪心插入所有弹射任务
        """
        new_routes = self._copy_routes(routes)
        self._remove_task(new_routes, sid_from, tid)
        task = self.tasks.get(tid)
        if task is None:
            return None

        # 收集目标船上容易弹射的任务 (按插入难度排序)
        target_tasks = []
        for rn in new_routes.get(sid_to, []):
            if rn.action == "PICKUP":
                target_tasks.append(rn.task_id)

        # 弹射策略: 随机选 depth 个任务从目标船弹射
        n_eject = min(depth, len(target_tasks))
        if n_eject == 0 and len(target_tasks) == 0:
            # 空船, 直接试试看
            pu, de, ok = self._find_insertion(self.ships[sid_to], new_routes[sid_to], task)
            if ok:
                self._do_insert(new_routes, sid_to, tid, pu, de, task)
                return new_routes, 0
            return None

        ejected = random.sample(target_tasks, n_eject)
        ejected_tasks_info = []
        for etid in ejected:
            t = self.tasks.get(etid)
            if t:
                ejected_tasks_info.append((etid, t))
            self._remove_task(new_routes, sid_to, etid)

        # 插入目标任务 tid 到目标船
        pu, de, ok = self._find_insertion(self.ships[sid_to], new_routes[sid_to], task)
        if not ok:
            # 弹射后仍不可行, 恢复原始
            return None

        self._do_insert(new_routes, sid_to, tid, pu, de, task)

        # 重新插入弹射的任务 (贪心, 不限于原船)
        all_inserted = True
        for etid, etask in ejected_tasks_info:
            best_sid, best_pu, best_de, best_score = None, -1, -1, float('inf')
            for sid in self.ships:
                ship = self.ships[sid]
                pu2, de2, ok2 = self._find_insertion(ship, new_routes[sid], etask)
                if not ok2:
                    continue
                dt = self._insertion_time_delta(ship, new_routes[sid], pu2, de2, etask)
                st = self._ship_time(ship, new_routes[sid])
                score = dt + 0.5 * st
                if score < best_score:
                    best_score = score
                    best_sid, best_pu, best_de = sid, pu2, de2

            if best_sid is not None:
                self._do_insert(new_routes, best_sid, etid, best_pu, best_de, etask)
            else:
                all_inserted = False
                break

        if not all_inserted:
            return None

        if not self._validate_all(new_routes):
            return None

        return new_routes, n_eject + 1  # ejected tasks + target task

    def _post_optimize(self, routes, n_iter=50):
        """最终微调: 局部搜索降低 makespan"""
        best = deepcopy(routes)
        best_ms = self.makespan(best)

        for _ in range(n_iter):
            tasks = self._collect_all_tasks(best)
            if len(tasks) < 2:
                continue

            # 随机 swap 两个不同船的任务
            a = random.choice(tasks)
            b = random.choice([t for t in tasks if t[0] != a[0]])
            if not b:
                continue

            new_routes = self._copy_routes(best)
            self._remove_task(new_routes, a[0], a[1])
            self._remove_task(new_routes, b[0], b[1])

            ta, tb = self.tasks.get(a[1]), self.tasks.get(b[1])
            if ta is None or tb is None:
                continue

            ok1 = self._greedy_insert_one(new_routes, a[0], b[1])
            ok2 = self._greedy_insert_one(new_routes, b[0], a[1])
            if not (ok1 and ok2):
                continue
            if not self._validate_all(new_routes):
                continue

            ms = self.makespan(new_routes)
            if ms < best_ms:
                best = new_routes
                best_ms = ms

        return best

    def _greedy_insert_one(self, routes, sid, tid):
        task = self.tasks.get(tid)
        if task is None:
            return False
        pu, de, ok = self._find_insertion(self.ships[sid], routes[sid], task)
        if ok:
            self._do_insert(routes, sid, tid, pu, de, task)
        return ok
