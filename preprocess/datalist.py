import os, argparse

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('-d','--dataset',required=True,type=str,choices=['GID','DeepGlobe'])
    parse.add_argument('-i','--imgpath',required=True,type=str)
    parse.add_argument('-l','--labpath',required=True,type=str)
    parse.add_argument('-s','--savepath',required=True,type=str)
    parse.add_argument('-r','--list',type=str)
    return parse.parse_args()

if __name__=='__main__':
    args = parse_args()
    img_list = os.listdir(args.imgpath)
    lab_list = os.listdir(args.labpath)
    assert len(img_list) == len(lab_list)
    img_list.sort()
    lab_list.sort()
    if args.list is not None:
        ref = []
        f = open(args.list,'r')
        line = f.readline()
        while line:
            ref.append(line.split('_')[0])
            line = f.readline()
        f.close()
    f_save = open(args.savepath,'w')
    for i in range(len(img_list)):
        assert os.path.splitext(img_list[i])[0] in lab_list[i]
        if (args.list is not None) and (img_list[i].split('_')[0] not in ref):
            continue
        f_save.write(img_list[i]+'\t'+lab_list[i]+'\n')
    f_save.close()