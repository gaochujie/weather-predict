from pypinyin import pinyin, lazy_pinyin, Style


def change(spinyin):
    string=pinyin(spinyin,heteronym=False,style=Style.NORMAL,strict=False)
    ss=""
    for i in range(0,len(string)):
        ss+=str(string[i][0])
    return ss
