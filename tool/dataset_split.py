import os, random, shutil
import argparse

def moveFile(imgDir,labelDir):
        ImgPathDir = os.listdir(imgDir)     
        labelPathDir = os.listdir(labelDir) 
        filenumber=len(ImgPathDir)          

        rate=0.2    
        picknumber=int(filenumber*rate) 
        sample = random.sample(ImgPathDir, picknumber)  
        print(sample)
        for name in sample:
                shutil.move(fileDir+name, tarDir+name)    
                try: 
                    shutil.move((labelfileDir+name).replace(".jpg",".def"), (labeltarDir+name).replace(".jpg",".def"))
                except:
                    continue
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-image', '--image', required=True, help='path to train dataset')
    parser.add_argument('-label', '--label', required=True, help='path to val dataset')
    parser.add_argument('-val', '--val', required=True, help='path to dest dataset')
    args = parser.parse_args()
    fileDir = args.image +"/"       
    tarDir = args.val +"/"        
    labelfileDir = args.label +"/"  
    labeltarDir = args.val +"/"
    moveFile(fileDir, labelfileDir)