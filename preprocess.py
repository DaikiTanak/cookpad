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
    #train_ing = ing
    train_title = []
    train_ing = []
    train_step = []
    for ing_list in ing:
        recipe_ing = []
        for i in ing_list:
            exp = m.parse(i).split(" ")[:-1]
            #print(exp)
            recipe_ing.extend(exp)
        train_ing.append(recipe_ing)
    for t in title:
        exp = m.parse(t).split(" ")[:-1]
        train_title.append(exp)

    for s_list in step:
        recipe_step = []
        for s in s_list:
            exp = m.parse(s).split(" ")[:-1]
            recipe_step.extend(exp)
        train_step.append(recipe_step)


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
    #test_ing = ing
    test_title = []
    test_ing = []
    test_step = []



    for ing_list in ing:
        recipe_ing = []
        for i in ing_list:
            exp = m.parse(i).split(" ")[:-1]
            #print(exp)
            recipe_ing.extend(exp)
        test_ing.append(recipe_ing)

    for t in title:
        #\nの除去
        exp = m.parse(t).split(" ")[:-1]
        test_title.append(exp)

    for s_list in step:
        recipe_step = []
        for s in s_list:
            exp = m.parse(s).split(" ")[:-1]
            recipe_step.extend(exp)
        test_step.append(recipe_step)
    print(test_step[:2])

    print(len(train_title))

    #train_title = train_title[:8000]
    print()
    return train_title, train_ing, train_step, test_title, test_ing, test_step

if __name__ == "__main__":
    title_ing()
