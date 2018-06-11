import MeCab
import sys

def title_ing():
    f = open('train.txt')
    train = f.read()  # ファイル終端まで全て読んだデータを返す
    f.close()
    f = open('test.txt')
    test = f.read()  # ファイル終端まで全て読んだデータを返す
    f.close()

    m = MeCab.Tagger("-Owakati")
    train_recipe = train.split("\n")
    test_recipe = test.split("\n")

    if_title, if_ing, if_step = False, False, False
    title, ing, step = [], [], []
    ing_list, step_list = [], []
    this_index = 0
    for row in train_recipe:

        if(if_title):
            title.append(row)
            if_title = False
        elif(if_ing):
            ing_list.append(row.split("\t")[0])
            if(train_recipe[this_index + 1] == "--- Step ---"):
                ing.append(ing_list)
                ing_list = []
                if_ing = False

        elif(if_step):
            step_list.append(row)
            if(train_recipe[this_index + 1] == "--- End ---"):
                step.append(step_list)
                step_list = []
                if_step= False

        if(row == "--- Title ---"):
            if_title = True
        elif(row == "--- Ingredients ---"):
            if_ing = True
        elif(row == "--- Step ---"):
            if_step = True

        this_index += 1
    print("training dataset done.")
    train_ing = ing
    train_title = []
    for t in title:
        exp = m.parse(t).split(" ")
        train_title.append(exp)

    if_title, if_ing, if_step = False, False, False
    title, ing, step = [], [], []
    ing_list, step_list = [], []
    this_index = 0
    for row in test_recipe:

        if(if_title):
            title.append(row)
            if_title = False
        elif(if_ing):
            ing_list.append(row.split("\t")[0])
            if(test_recipe[this_index + 1] == "--- Step ---"):
                ing.append(ing_list)
                ing_list = []
                if_ing = False

        elif(if_step):
            step_list.append(row)
            if(test_recipe[this_index + 1] == "--- End ---"):
                step.append(step_list)
                step_list = []
                if_step= False

        if(row == "--- Title ---"):
            if_title = True
        elif(row == "--- Ingredients ---"):
            if_ing = True
        elif(row == "--- Step ---"):
            if_step = True

        this_index += 1
    print("test dataset done.")
    test_ing = ing
    test_title = []
    for t in title:
        exp = m.parse(t).split(" ")
        test_title.append(exp)

    return train_title, train_ing, test_title, test_ing
