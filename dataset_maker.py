import pickle
import numpy as np
import math


def make_dataset_forcemeals(user_daily_meals, parts=3, days=3, only_major=False):
    final_dataset = False
    timeband = 1 / parts
    count = 0
    if only_major:
        nut_num = 4
    else:
        nut_num = 31
    for user_id, daily_meals in user_daily_meals.items():
        meals_list = [False] * days
        for day, meals in sorted(daily_meals.items()):
            meals_today = np.asarray([np.zeros(nut_num)] * parts)
            for time, meal in meals:
                if only_major:
                    meal = meal[:4]
                index = math.floor(time / timeband)
                if index == parts:
                    index = 0
                for i in range(len(meal)):
                    if i == 0:
                        continue
                    meal[i] /= meal[0]
                meals_today[index] += np.asarray(meal)
                meals_list[0] = [day, meals_today]

            for i in range(1, days):
                if meals_list[i]:
                    meals_list[i][1] = np.concatenate((meals_list[i][1], meals_list[0][1]), axis=0)
                else:
                    meals_list[i] = meals_list[0]

            if meals_list[days-1][1].shape[0] == parts*days:
                count+=1
                if type(final_dataset) != list:
                    final_dataset = [(user_id, meals_list[days-1][0], meals_list[days-1][1])]
                else:
                    final_dataset.append((user_id, meals_list[days-1][0], meals_list[days-1][1]))

            meals_list = meals_list[:-1]
            meals_list.insert(0, False)
    print("day: ", days, " parts: ", parts, " num: ", len(final_dataset))

    with open("data/subdata/user_meals_dataset_FM_days_" + str(days) + "_parts_" + str(parts) + "_nut_" + str(nut_num) + "_balance.p", "wb") as f:
        pickle.dump(final_dataset, f)

if __name__ == '__main__':
    with open('data/subdata/user_daily_meals.p', mode='rb') as f:
        user_daily_meals = pickle.load(f)
    parts = [1, 3, 6, 8]
    days = [1, 3, 7]
    for part in parts:
        for day in days:
            make_dataset_forcemeals(user_daily_meals, part, day, False)
            make_dataset_forcemeals(user_daily_meals, part, day, True)