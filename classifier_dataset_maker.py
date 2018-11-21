import pickle
import numpy as np
import math


def make_dataset_forcemeals(user_daily_meals, train_id, parts=3, days=3, only_major=False):
    final_train_dataset = False
    final_val_dataset = False
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
                if meal[0] > 0:
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
                if user_id in train_id:
                    if type(final_train_dataset) != list:
                        final_train_dataset = [(user_id, meals_list[days-1][0], meals_list[days-1][1])]
                    else:
                        final_train_dataset.append((user_id, meals_list[days-1][0], meals_list[days-1][1]))
                else:
                    if type(final_val_dataset) != list:
                        final_val_dataset = [(user_id, meals_list[days-1][0], meals_list[days-1][1])]
                    else:
                        final_val_dataset.append((user_id, meals_list[days-1][0], meals_list[days-1][1]))

            meals_list = meals_list[:-1]
            meals_list.insert(0, False)
    print("day: ", days, " parts: ", parts, " train num: ", len(final_train_dataset), " val num: ", len(final_val_dataset))

    with open("data/subdata/classifier/" + str(days) + "_days_" + str(parts) + "_parts_" + str(nut_num) + "_train.p", "wb") as f:
        pickle.dump(final_train_dataset, f)
    with open("data/subdata/classifier/" + str(days) + "_days_" + str(parts) + "_parts_" + str(nut_num) + "_val.p", "wb") as f:
        pickle.dump(final_val_dataset, f)

if __name__ == '__main__':
    train_max = 100
    with open('data/subdata/user_daily_meals.p', mode='rb') as f:
        user_daily_meals = pickle.load(f)
    with open('data/subdata/serve_by_user/user_ids.p', mode='rb') as f:
        user_ids = pickle.load(f)
    train_id = []
    val_id = []
    for user_id in user_ids:
        if np.random.rand() > 0.8 and len(train_id) < 100:
            train_id.append(user_id)
        else:
            val_id.append(user_id)
    print(len(train_id), len(val_id))

    parts = [1, 3, 6, 8]
    days = [1, 3, 7]
    for part in parts:
        for day in days:
            make_dataset_forcemeals(user_daily_meals, train_id, part, day, False)
            make_dataset_forcemeals(user_daily_meals, train_id, part, day, True)