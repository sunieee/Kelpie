import numpy as np
from gurobipy import Model, GRB, quicksum
import json
import math


def optimize_index(facts, gamma = 0.1):
    # 步骤1：数据解析和预处理
    N = len(facts)  # 事实数量
    print('# of facts:', N) 
    # print([fact['triple' ] for fact in facts])

    # 提取每个事实的得分
    Score = np.array([fact['score'] for fact in facts])

    # 构建规则集合和规则权重
    # 创建一个字典，存储每个规则的权重w(R)
    rule_weights = {}  # key: rule_id, value: w(R)

    # 创建一个字典，存储每个事实包含的规则集合
    fact_rules = []  # 每个元素是一个集合，表示该事实包含的规则ID

    # 首先遍历所有事实，收集规则信息
    for fact in facts:
        rules = fact['rules'] if 'rules' in fact else []
        rule_ids = set()
        for rule in rules:
            rule_id = rule['id']
            # 计算w(R)
            if rule_id not in rule_weights:
                rule_length = len(rule_id.split(','))
                w_R = rule['SC'] / rule_length
                rule_weights[rule_id] = w_R
            rule_ids.add(rule_id)
        fact_rules.append(rule_ids)

    # 构建phi(f_i, R)的字典
    # key: (fact_index, rule_id), value: phi(f_i, R)
    phi = {}

    for i, fact in enumerate(facts):
        rules = fact['rules'] if 'rules' in fact else []
        for rule in rules:
            rule_id = rule['id']
            importance = rule['importance']
            phi[(i, rule_id)] = importance

    # 构建关联度矩阵Rel
    Rel = np.zeros((N, N))

    for i in range(N):
        for j in range(i+1, N):
            # 找到共同的规则
            common_rules = fact_rules[i].intersection(fact_rules[j])
            if common_rules:
                rel_ij = 0
                for rule_id in common_rules:
                    w_R = rule_weights[rule_id]
                    phi_i_R = phi[(i, rule_id)]
                    phi_j_R = phi[(j, rule_id)]
                    rel_ij += w_R * math.sqrt(phi_i_R * phi_j_R)
                Rel[i, j] = rel_ij
                Rel[j, i] = rel_ij  # 确保对称

    # 步骤2：构建优化模型
    model = Model('SelectFacts')

    # 添加决策变量
    x = model.addVars(N, vtype=GRB.BINARY, name="x")

    # 设置目标函数
    obj_linear = quicksum(x[i] * Score[i] for i in range(N))
    obj_quadratic = gamma * quicksum(
        x[i] * x[j] * Rel[i, j]
        for i in range(N) for j in range(i+1, N) if Rel[i, j] != 0
    )

    model.setObjective(obj_linear + obj_quadratic, GRB.MAXIMIZE)

    # 添加约束：选择4个事实
    model.addConstr(quicksum(x[i] for i in range(N)) == 4, name="Select4")

    # 可选：设置求解参数
    model.Params.TimeLimit = 600  # 设置最大求解时间为600秒
    model.Params.MIPGap = 0.01    # 允许1%的最优性Gap
    model.Params.Threads = 4      # 使用4个线程

    # 步骤3：求解优化模型
    model.optimize()

    # 步骤4：输出结果
    
    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
        selected_indices = [i for i in range(N) if x[i].X > 0.5]
        value = model.ObjVal
        print("Selected facts indices:", selected_indices)
        print("Maximum Score:", model.ObjVal)
    else:
        selected_indices = list(range(N)) if N < 4 else [0, 1, 2, 3]
        value = np.sum([Score[i] for i in selected_indices])
        print("No feasible solution found within the time limit.")

    return selected_indices, value


if __name__ == '__main__':
    with open('out/complex_FB15k-237/extractedFactsMap.json', 'r') as f:
        data = json.load(f)

    facts = list(data.values())[0]

    print(facts[0])
    for facts in data.values():
        optimize_index(facts)