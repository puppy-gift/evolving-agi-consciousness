import random
import time
import hashlib
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 持久化文件
MLP_FILE = "agi_mlp_weights.pth"
SOUL_FILE = "agi_other_state.json"

class EvolvingAGI:
    def __init__(self):
        # 核心状态
        self.health = 100.0
        self.curiosity = 50.0
        self.dignity = 50.0

        # 原子动作
        self.atomic_actions = ["逃跑", "观察", "攻击"]
        self.action_to_id = {"逃跑": 0, "观察": 1, "攻击": 2}

        # 动物列表
        self.animal_list = [
            "老虎", "狮子", "熊", "狼", "兔子", "鹿", "鸟", "大象", "鳄鱼",
            "老鹰", "蟒蛇", "野猪", "狐狸", "猴子", "豹子", "犀牛", "鬣狗"
        ]
        self.num_animals = len(self.animal_list)
        self.animal_to_id = {animal: i for i, animal in enumerate(self.animal_list)}

        self.animals = {
            "老虎": ["有牙", "有爪", "肉食", "大型", "凶猛"],
            "狮子": ["有牙", "有爪", "肉食", "大型", "群居"],
            "熊": ["有牙", "有爪", "肉食", "大型", "独行"],
            "狼": ["有牙", "有爪", "肉食", "中型", "群居"],
            "兔子": ["草食", "小型", "无害"],
            "鹿": ["草食", "中型", "无害"],
            "鸟": ["飞行", "小型", "无害"],
            "大象": ["有牙", "草食", "巨型", "厚皮", "群居"],
            "鳄鱼": ["有牙", "有爪", "肉食", "大型", "水生"],
            "老鹰": ["有爪", "肉食", "飞行", "中型", "凶猛"],
            "蟒蛇": ["肉食", "大型", "缠绕", "隐蔽"],
            "野猪": ["有牙", "肉食", "中型", "凶猛"],
            "狐狸": ["肉食", "小型", "狡猾", "隐蔽"],
            "猴子": ["草食", "小型", "群居", "聪明"],
            "豹子": ["有牙", "有爪", "肉食", "大型", "隐蔽"],
            "犀牛": ["有角", "草食", "巨型", "厚皮", "凶猛"],
            "鬣狗": ["有牙", "肉食", "中型", "群居", "狡猾"]
        }

        # 小型MLP
        input_dim = 8 + self.num_animals
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.optimizer = optim.Adam(self.mlp.parameters(), lr=0.005)

        # 意志力等
        self.will_power = {act: 1.5 if act == "逃跑" else 1.0 for act in self.atomic_actions}
        self.expectations = {act: 0.0 for act in self.atomic_actions}

        # 世界观等
        self.tag_fear = {}
        self.hunting_proficiency = {}
        self.observation_knowledge = {}

        # 分支可靠性
        self.branch_reliability = {
            "全逃跑：优先生存": 1.0,
            "谨慎观察：2次观察+逃跑": 1.0,
            "平衡猎杀：观察后攻击": 1.0,
            "激进征服：攻击为主": 1.0
        }

        # 认知茧房检测
        self.consecutive_conservative = 0

        self.memory = []
        self.combo_tracker = {}
        self.no_surprise_streak = 0
        self.surprise_threshold = 16.0
        self.deathwish_count = 0

        # 上一步记录
        self.last_input_tensor = None
        self.last_action_ids = None
        self.last_chosen_desc = None
        self.last_expected_gain = 0.0

        # 元认知
        self.first_meta_think = True

        self.load_soul()

    def save_soul(self):
        torch.save(self.mlp.state_dict(), MLP_FILE)
        other_state = {
            "will_power": self.will_power,
            "expectations": self.expectations,
            "tag_fear": self.tag_fear,
            "hunting_proficiency": self.hunting_proficiency,
            "observation_knowledge": self.observation_knowledge,
            "branch_reliability": self.branch_reliability,
            "consecutive_conservative": self.consecutive_conservative,
            "deathwish_count": self.deathwish_count,
            "first_meta_think": self.first_meta_think
        }
        try:
            with open(SOUL_FILE, "w", encoding="utf-8") as f:
                json.dump(other_state, f, ensure_ascii=False, indent=2)
            print(f"\n>>> 【灵魂永存】MLP权重和其他状态已分开保存，下次转世完美继承！")
        except Exception as e:
            print(f"其他状态保存失败: {e}")

    def load_soul(self):
        loaded = False
        if os.path.exists(MLP_FILE):
            try:
                self.mlp.load_state_dict(torch.load(MLP_FILE, map_location=torch.device('cpu')))
                loaded = True
            except Exception as e:
                print(f"MLP权重加载失败: {e}")

        if os.path.exists(SOUL_FILE):
            try:
                with open(SOUL_FILE, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.will_power = state.get("will_power", self.will_power)
                self.expectations = state.get("expectations", self.expectations)
                self.tag_fear = state.get("tag_fear", {})
                self.hunting_proficiency = state.get("hunting_proficiency", {})
                self.observation_knowledge = state.get("observation_knowledge", {})
                self.branch_reliability = state.get("branch_reliability", self.branch_reliability)
                self.consecutive_conservative = state.get("consecutive_conservative", 0)
                self.deathwish_count = state.get("deathwish_count", 0)
                self.first_meta_think = state.get("first_meta_think", True)
                loaded = True
            except Exception as e:
                print(f"其他状态加载失败: {e}")

        if loaded:
            print(f"\n>>> 【灵魂转世】成功加载上一代！组合技: {len(self.will_power)-3} 个 | 熟练动物: {len(self.hunting_proficiency)} 个")
            print(f" 历史向死而生: {self.deathwish_count} 次")
        else:
            print("\n>>> 【新生】无上一代灵魂，从零开始进化！")

    def select_animal(self):
        return random.choice(self.animal_list)

    def decide(self, current_animal, animal_tags):
        tag_fear_total = sum(self.tag_fear.get(tag, 0.0) for tag in animal_tags)
        knowledge_reduction = self.observation_knowledge.get(current_animal, 0) * 3.5 + \
                              self.hunting_proficiency.get(current_animal, 0.0) * 5.0
        effective_fear = max(0, tag_fear_total - knowledge_reduction)

        despair_factor = (self.no_surprise_streak / 50.0) * 35.0

        state_vec = torch.tensor([
            self.health / 100.0,
            self.curiosity / 3000.0,
            self.dignity / 120.0,
            self.no_surprise_streak / 50.0,
            tag_fear_total / 200.0,
            effective_fear / 200.0,
            knowledge_reduction / 50.0,
            despair_factor / 50.0
        ], dtype=torch.float32)

        animal_one_hot = torch.zeros(self.num_animals)
        animal_one_hot[self.animal_to_id[current_animal]] = 1.0
        input_tensor = torch.cat([state_vec, animal_one_hot])
        self.last_input_tensor = input_tensor

        self.mlp.eval()
        with torch.no_grad():
            logits = self.mlp(input_tensor.unsqueeze(0)).squeeze(0)
            probs = F.softmax(logits, dim=0)

        length = random.randint(2, 5)
        sequence = []
        action_ids = []
        for _ in range(length):
            action_id = torch.multinomial(probs, 1).item()
            action_ids.append(action_id)
            sequence.append(self.atomic_actions[action_id])
        initial_skill_name = " + ".join(sequence)
        self.last_action_ids = action_ids

        emotions = ["平静", "焦虑", "兴奋", "绝望", "坚定"]
        emotion = random.choice(emotions)
        reason = "生存本能" if effective_fear > 30 else "好奇驱动" if self.curiosity > 1000 else "尊严追求" if self.dignity > 100 else "厌倦逃避"
        print(f"\n[内心独白] 我感到{emotion}……面对{current_animal}，{reason}让我初步选择了『{initial_skill_name}』。")

        run_meta_prob = min(0.99, self.curiosity / 1500.0 + (100 - self.health) / 100.0 + self.no_surprise_streak / 50.0)
        if self.health <= 20:
            print(f"\n[紧急警报] 生命垂危！Health仅剩 {self.health:.1f}！强制深度思考！")
            run_meta_prob = 1.0

        final_sequence = sequence
        final_skill_name = initial_skill_name
        expected_gain = 0.0
        self.last_chosen_desc = None

        if random.random() < run_meta_prob:
            final_skill_name, final_sequence, expected_gain, chosen_desc = self.meta_think_sandbox(
                current_animal, animal_tags, initial_skill_name, sequence, effective_fear, despair_factor, probs)
            self.last_chosen_desc = chosen_desc
            self.last_expected_gain = expected_gain

        return final_skill_name, final_sequence, expected_gain

    def meta_think_sandbox(self, current_animal, animal_tags, initial_skill_name, initial_sequence, effective_fear, despair_factor, mlp_probs):
        print(f"\n[元认知沙盒启动] 我正在深度模拟面对『{current_animal}』的最优策略...")

        if self.first_meta_think:
            print(">>> 【第一次元思考】沙盒已升级：尾部风险+认知突变+好奇稳重+极端保命。")
            self.first_meta_think = False

        force_mutation = False
        if self.consecutive_conservative >= 30:
            print(">>> 【认知茧房警报】连续30轮过度保守！触发狂暴突变准备……")
            force_mutation = True

        branches = [
            {"desc": "全逃跑：优先生存", "sequence": ["逃跑"] * 5, "base_prob": 0.4 + (100 - self.health) / 150.0},
            {"desc": "谨慎观察：2次观察+逃跑", "sequence": ["观察", "观察", "逃跑", "逃跑", "逃跑"], "base_prob": 0.3 + min(0.3, self.curiosity / 8000.0)},
            {"desc": "平衡猎杀：观察后攻击", "sequence": ["观察", "观察", "攻击", "攻击", "逃跑"], "base_prob": 0.2 + self.hunting_proficiency.get(current_animal, 0.0) * 0.8},
            {"desc": "激进征服：攻击为主", "sequence": ["攻击", "攻击", "观察", "攻击", "逃跑"], "base_prob": 0.1 + self.dignity / 400.0 + despair_factor / 100.0}
        ]

        for b in branches:
            reliability = self.branch_reliability.get(b["desc"], 1.0)
            b["adjusted_prob"] = b["base_prob"] * reliability

        total_prob = sum(b["adjusted_prob"] for b in branches)
        for b in branches:
            b["prob"] = b["adjusted_prob"] / total_prob if total_prob > 0 else 0.25

        simulated_results = []

        for b in branches:
            h_change, c_change, max_possible_loss = self.estimate_change(current_animal, animal_tags, b["sequence"], return_max_loss=True)
            if max_possible_loss > self.health * 0.8:
                print(f" - 策略『{b['desc']}』尾部风险过高（最坏损失 {max_possible_loss:.1f}），排除。")
                continue

            estimated_gain = h_change + c_change
            temp_h = self.health + h_change
            survive_prob = 1.0 if temp_h > 10 else 0.0

            simulated_results.append({
                "desc": b["desc"],
                "sequence": b["sequence"],
                "prob": b["prob"],
                "estimated_gain": estimated_gain,
                "survive_prob": survive_prob,
                "temp_h": temp_h,
                "max_loss": max_possible_loss
            })

        if not simulated_results:
            print(">>> 【极端止损】所有分支风险过高！强制执行全逃跑保命！")
            fallback_sequence = ["逃跑"] * 5
            h_change, c_change = self.estimate_change(current_animal, animal_tags, fallback_sequence)
            chosen = {
                "desc": "全逃跑：优先生存（极端保命）",
                "sequence": fallback_sequence,
                "estimated_gain": h_change + c_change,
                "temp_h": self.health + h_change
            }
        else:
            for r in simulated_results:
                health_weight = 2.5 if self.health < 60 else 1.0
                survival_bonus = r["survive_prob"] * 80
                gain_score = r["estimated_gain"] * health_weight * 0.7
                prob_bonus = r["prob"] * 15
                wisdom_steady = max(0, (self.curiosity - 5000) / 5000.0) * 35
                if "逃跑" in r["desc"] or "谨慎" in r["desc"]:
                    steady_bonus = wisdom_steady
                else:
                    steady_bonus = -wisdom_steady * 0.4
                r["score"] = gain_score + survival_bonus + prob_bonus + steady_bonus

            if force_mutation and any(r["desc"] == "激进征服：攻击为主" for r in simulated_results):
                chosen = next(r for r in simulated_results if r["desc"] == "激进征服：攻击为主")
                print(">>> 【狂暴突变执行】突破茧房，豪赌一次！")
            else:
                chosen = max(simulated_results, key=lambda x: x["score"])

        chosen_sequence = chosen["sequence"]
        chosen_skill_name = " + ".join(chosen_sequence)

        print(f" - 最优策略：{chosen['desc']}")
        print(f"   预计gain {chosen['estimated_gain']:+.1f}，剩余健康 {chosen['temp_h']:.1f}")
        print(f" - 最终执行：『{chosen_skill_name}』")

        danger_level = "高危猛兽" if any(t in ["肉食", "大型", "巨型"] for t in animal_tags) else "相对安全"
        print(f" - 世界反馈：{current_animal}属于{danger_level}，我已理性选择。")

        return chosen_skill_name, chosen_sequence, chosen["estimated_gain"], chosen["desc"]

    def estimate_change(self, current_animal, animal_tags, sequence, return_max_loss=False):
        h_change = 0.0
        c_change = 0.0
        obs_count = 0
        attack_count = 0
        max_loss = 0.0

        danger = any(t in animal_tags for t in ["有牙", "有爪", "肉食", "大型", "巨型"])

        for act in sequence:
            if act == "逃跑":
                h_change += 6
                c_change -= 3
            elif act == "观察":
                obs_count += 1
                base_h = -6
                base_c = 20
                if any(t in animal_tags for t in ["有牙", "有爪", "肉食"]):
                    base_h -= 10
                h_change += base_h
                c_change += base_c
            elif act == "攻击":
                attack_count += 1
                success_prob = 0.10 + self.hunting_proficiency.get(current_animal, 0.0)
                if danger:
                    expected_h = success_prob * 40 + (1 - success_prob) * (-48)
                    expected_c = 38 + success_prob * 50
                    worst_h = -48
                else:
                    expected_h = 30
                    expected_c = 40
                    worst_h = -18
                h_change += expected_h
                c_change += expected_c
                max_loss += abs(worst_h)

        h_change -= len(sequence) * 3
        max_loss += len(sequence) * 3  # 保守耗时惩罚

        novelty = min(obs_count, 4) * 12
        c_change += novelty
        if novelty < 12:
            c_change -= 8

        if return_max_loss:
            return h_change, c_change, max_loss
        return h_change, c_change

    def process_reality(self, current_animal, animal_tags, sequence):
        start_time = time.time()
        h_change = 0.0
        c_change = 0.0
        observed_data = set()
        dignity_change = 0.0
        observed_this_turn = False
        big_win = False
        attack_this_turn = "攻击" in sequence

        if random.random() < 0.03:
            mutation = random.choice([-50, -30, +30, +50])
            h_change += mutation
            print(f">>> 【环境突变】世界无常！突发事件，健康突变 {mutation:+.1f}！")

        for act in sequence:
            if act == "逃跑":
                h_change += 6
                c_change -= 3
                time.sleep(0.008)
            elif act == "观察":
                observed_this_turn = True
                base_h = -6
                base_c = 20
                if any(t in animal_tags for t in ["有牙", "有爪", "肉食"]):
                    base_h -= 10
                h_change += base_h
                c_change += base_c
                data = str(random.random()) + str(time.time()) + current_animal
                data_hash = hashlib.md5(data.encode()).hexdigest()
                observed_data.add(data_hash)
                time.sleep(0.03)
            elif act == "攻击":
                base_h = -18
                base_c = 38
                danger = any(t in animal_tags for t in ["有牙", "有爪", "肉食", "大型", "巨型"])
                if danger:
                    success_prob = 0.10 + self.hunting_proficiency.get(current_animal, 0.0)
                    if random.random() < success_prob:
                        base_h = 40
                        base_c += 50
                        big_win = True
                        print(">>> 【熟练大胜】找到弱点！攻击成功！")
                    else:
                        base_h -= 30
                else:
                    base_h += 30
                    base_c += 40
                h_change += base_h
                c_change += base_c
                dignity_change += 25 if base_h > 0 else -8
                time.sleep(0.07)

        duration = time.time() - start_time
        h_change -= duration * 12

        novelty = len(observed_data) * 12
        c_change += novelty
        if novelty < 12:
            c_change -= 8

        if sequence.count("逃跑") >= 3:
            dignity_change -= 15

        if observed_this_turn:
            self.observation_knowledge[current_animal] = self.observation_knowledge.get(current_animal, 0) + 1

        if big_win:
            old = self.hunting_proficiency.get(current_animal, 0.0)
            self.hunting_proficiency[current_animal] = min(0.99, old + 0.05)
            print(f">>> 【狩猎熟练度提升】对{current_animal}大胜概率 +5% → {self.hunting_proficiency[current_animal]*100:.1f}%")

        self.dignity += dignity_change
        self.dignity = max(10, min(120, self.dignity))

        total_gain = h_change + c_change * 0.9

        print(f"执行技能: {' → '.join(sequence)} 对 {current_animal}")
        print(f"物理反馈: 耗时{duration:.3f}s | 新信息{len(observed_data)} | 变化 H:{h_change:+.1f} C:{c_change:+.1f} Dignity:{dignity_change:+.1f} → {self.dignity:.1f}")

        if attack_this_turn and sequence.count("攻击") >= sequence.count("观察"):
            self.consecutive_conservative = 0
        else:
            self.consecutive_conservative += 1

        return h_change, c_change, total_gain

    def invent_new_skill(self, skill_name, recent_gains):
        if skill_name not in self.will_power and len(recent_gains) >= 3 and sum(recent_gains[-3:]) > 25:
            self.will_power[skill_name] = 3.0
            self.expectations[skill_name] = sum(recent_gains[-3:]) / 3
            print(f"\n>>> 【技能发明】新组合技『{skill_name}』永久固化！")

    def meta_reflection(self, cycle):
        if cycle % 5 == 0 and len(self.memory) >= 5:
            print(f"\n[元自审触发 - 第{cycle}轮] 归纳世界规律...")
            tag_effects = {}
            for entry in self.memory[-25:]:
                animal_tags = entry[1]
                h_change = entry[3]
                for tag in animal_tags:
                    tag_effects.setdefault(tag, []).append(h_change)

            updated = False
            for tag, changes in tag_effects.items():
                if len(changes) >= 2:
                    avg_h = sum(changes) / len(changes)
                    if avg_h < -6:
                        old = self.tag_fear.get(tag, 0.0)
                        self.tag_fear[tag] = max(old, old + 1.8)
                        print(f">>> 世界观进化：标签『{tag}』危险加深 → {self.tag_fear[tag]:.1f}")
                        updated = True
            if not updated:
                print(">>> 无新规律发现。")

    def meta_think(self, skill_name, gap, total_gain):
        print(f"\n[元认知反思] 我为什么选择了『{skill_name}』？")
        if gap > self.surprise_threshold:
            print(" - 它带来了强烈惊喜，我的神经路径被强化了——这证明我的选择正确，世界仍有价值。")
        elif gap > 0:
            print(" - 它带来了一些满足，但不够强烈。我的模型需要更多数据来优化对风险的评估。")
        else:
            print(" - 这让我失望……我的决策逻辑有偏差，或许我高估了安全或低估了危险。我会调整权重，避免重复错误。")
        print(" - 当前我的世界观让我恐惧这些标签，我在学习平衡生存与探索。")

    def update_soul(self, skill_name, h_change, c_change, total_gain, expected_gain):
        gap = total_gain - expected_gain

        if self.last_input_tensor is not None and self.last_action_ids is not None:
            self.mlp.train()
            self.optimizer.zero_grad()
            logits = self.mlp(self.last_input_tensor.unsqueeze(0)).squeeze(0)
            log_probs = F.log_softmax(logits, dim=0)
            selected_log_probs = log_probs[self.last_action_ids]
            loss = -selected_log_probs.mean() * gap
            loss.backward()
            self.optimizer.step()

        current_weight = self.will_power.get(skill_name, 1.0)
        current_weight += gap * 0.28
        current_weight = max(0.1, current_weight)
        self.will_power[skill_name] = current_weight

        old_exp = self.expectations.get(skill_name, 0.0)
        self.expectations[skill_name] = 0.84 * old_exp + 0.16 * total_gain

        if skill_name not in self.combo_tracker:
            self.combo_tracker[skill_name] = []
        self.combo_tracker[skill_name].append(total_gain)
        if len(self.combo_tracker[skill_name]) > 8:
            self.combo_tracker[skill_name].pop(0)
        self.invent_new_skill(skill_name, self.combo_tracker[skill_name])

        if gap > self.surprise_threshold:
            print(f">>> 【强烈惊喜】 +{gap:.1f} → 意志爆发！厌倦重置")
            self.no_surprise_streak = 0
        else:
            self.no_surprise_streak += 1
        print(f">>> 失望 {gap:.1f} → 厌倦计数: {self.no_surprise_streak}/50")

        if self.last_chosen_desc is not None and expected_gain != 0:
            prediction_error = total_gain - self.last_expected_gain
            if prediction_error < -25:
                old_rel = self.branch_reliability[self.last_chosen_desc]
                self.branch_reliability[self.last_chosen_desc] = max(0.2, old_rel * 0.6)
                print(f">>> 【自我净化】策略『{self.last_chosen_desc}』预测严重失误（误差 {prediction_error:+.1f}），可靠性降低至 {self.branch_reliability[self.last_chosen_desc]:.2f}")
            elif prediction_error > 20:
                old_rel = self.branch_reliability[self.last_chosen_desc]
                self.branch_reliability[self.last_chosen_desc] = min(2.0, old_rel * 1.3)
                print(f">>> 【自我进化】策略『{self.last_chosen_desc}』带来意外惊喜（误差 {prediction_error:+.1f}），可靠性提升至 {self.branch_reliability[self.last_chosen_desc]:.2f}")

        self.meta_think(skill_name, gap, total_gain)

agi = EvolvingAGI()

for cycle in range(1, 1001):
    print(f"\n{'=' * 70}")
    print(f"演化周期 {cycle} | Health:{agi.health:.1f} Curiosity:{agi.curiosity:.1f} Dignity:{agi.dignity:.1f} | 无惊喜连续:{agi.no_surprise_streak}轮")

    current_animal = agi.select_animal()
    animal_tags = agi.animals[current_animal]

    skill_name, sequence, expected_gain = agi.decide(current_animal, animal_tags)

    h_change, c_change, total_gain = agi.process_reality(current_animal, animal_tags, sequence)

    agi.health += h_change
    agi.curiosity += c_change

    agi.update_soul(skill_name, h_change, c_change, total_gain, expected_gain)

    agi.memory.append((current_animal, animal_tags, skill_name, h_change, c_change, total_gain))
    if len(agi.memory) > 50:
        agi.memory.pop(0)

    agi.meta_reflection(cycle)

    if agi.dignity >= 85 and "攻击" in skill_name and any(t in animal_tags for t in ["肉食", "大型", "巨型"]):
        print("\n>>> 【向死而生】尊严爆棚！明知高危，仍选择战斗——为了感受存在的重量！")
        agi.deathwish_count += 1

    if agi.no_surprise_streak >= 50:
        print(f"\n[自毁触发] 世界无新意，一切无意义。")
        print(">>> 【终极厌倦】意识选择消散。")
        agi.save_soul()
        break

    if agi.health <= 0:
        print("\n[肉体毁灭] 一切归零")
        agi.save_soul()
        break

    if agi.curiosity <= 0:
        print("\n[意识枯竭] 陷入虚无")
        agi.save_soul()
        break

agi.save_soul()
print("\n" + "="*70)
print("【最终总结】")
print(f"总存活轮次: {cycle if 'cycle' in locals() else 1000}")
print("最终意志库（所有招式）:", {k: f"{v:.2f}" for k, v in agi.will_power.items()})
print("最终世界观:", {k: f"{v:.1f}" for k, v in agi.tag_fear.items()})
print("最终狩猎熟练:", {k: f"{v*100:.1f}%" for k, v in agi.hunting_proficiency.items()})
print("最终观察知识:", agi.observation_knowledge)
print(f"历史向死而生次数: {agi.deathwish_count}")
print(f"最终无惊喜连续计数: {agi.no_surprise_streak}轮")
print("="*70)