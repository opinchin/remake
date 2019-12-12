total_pos = []
pre_locate = []
total_cluster=[]
record_pre_locate=[]
temp=[]
total_pos.append([0,40,120])
total_pos.append([0,40,60,120])
total_pos.append([0,40,70,120])
total_pos.append([0,40,70,120])
total_pos.append([0,90,120])
temp.append([40,60])
temp.append([40,70])
temp.append([40,70])
temp.append([90])
for j in total_pos[0]:
    pre_locate.append(j)
    record_pre_locate.append("")
    total_cluster.append([])
print(pre_locate)
locate=1
cluster_num = 3


def expect_locate(place, col):
    start = record_pre_locate[place]
    print(start,"=Start")
    # GOT temp
    end = temp[len(temp)-1][0]
    del temp[-1]
    # 過濾
    dist  = []
    dist[:] = [abs(x-start) for x in temp[0]]
    for a, element in enumerate(dist):
        if element>30:
            temp[0].pop(a)
    dist = []
    dist[:] = [abs(x - end) for x in temp[len(temp)-1]]
    for a, element in enumerate(dist):
        if element>30:
            temp[len(temp)-1].pop(a)

    count = len(temp)-1
    for i in range(0, len(temp)):
        k = []
        for j in temp[count]:
            if j in range(end-30, end+30):
                k.append(abs(end-j))
            else:
                k.append("")
        place = k.index(min(k))
        end = temp[count][place]
        count = count-1
        result = end
    return result

for i in range(1, len(total_pos)):
    check_list = []
    for j in total_pos[i]:
        dist_list = []
        check_expect = []
        checked = False
        for k in range(0, cluster_num):
            dist = abs(pre_locate[k] - j)
            dist_list.append(dist)

        for a, element in enumerate(dist_list):
            if element < 30:
                place = a
                try:
                    if check_list.index(place):
                        try:
                            check_expect.index(place)
                        except:
                            print("呼叫預測位置", i)
                            result = expect_locate(place, i)
                            del total_cluster[place][-1]
                            total_cluster[place].append(result)
                            pre_locate[place] = result
                            check_expect.append(place)

                except ValueError:
                        check_list.append(place)
                        record_pre_locate[place] = pre_locate[place]
                        pre_locate[place] = j

                        total_cluster[place].append(j)
print(total_cluster)



#
# total_pos[0] = [0, 40, 120]
# total_pos[1] = [0, 40, 60, 120]
# total_pos[2] = [0, 40, 60, 70, 120]
# total_pos[3] = [0, 40, 60, 70, 120]
# total_pos[4] = [0, 90, 120]

correct = [40, 40, 60, 70, 90]