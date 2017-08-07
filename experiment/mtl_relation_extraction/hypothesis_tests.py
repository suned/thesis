import pandas
from scipy import stats


def hypothesis_tests(result1, result2):
    result1_name = result1.split("/")[-2]
    result2_name = result2.split("/")[-2]
    result1_data = pandas.read_csv(result1)
    result2_data = pandas.read_csv(result2)

    result1_f1 = result1_name + "_mean_f1"
    result2_f1 = result2_name + "_mean_f1"

    tests = {
        "fraction": [],
        result1_f1: [],
        result2_f1: [],
        "p-value": []
    }

    for (f, result1_group), (_, result2_group) in zip(result1_data.groupby("targetFraction"),
                                                      result2_data.groupby("targetFraction")):

        tests["fraction"].append(f)
        tests[result1_f1].append(result1_group.f1.mean())
        tests[result2_f1].append(result2_group.f1.mean())

        t_stat, p_value = stats.ttest_ind(result1_group.f1, result2_group.f1)
        tests["p-value"].append(p_value / 2)
    print(pandas.DataFrame(tests).transpose())
