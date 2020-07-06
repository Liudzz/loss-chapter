from PIL import Image
import glob

files = input("Please input the path of your first path:\n")
im = Image.open(files)
images = []
format = files.split('/')[-1].split(".")[-1]
imgspath = files.split('/')[0]
image = glob.glob(imgspath+"/*."+format)

for i in range(len(image)):
    if i!= 0:
        images.append(Image.open(imgspath+'/epoch_'+str(i+1)+'.png'))
        print(i)

im.save(imgspath+'.gif',"GIF",save_all=True,append_images=images,loop = 1,duration=0.001,comment=b'abcdefg')