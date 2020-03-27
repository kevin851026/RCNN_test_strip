from PIL import Image
import matplotlib.pyplot as plt
import os
import multiprocessing
#for i in os.listdir('.'):
#    im = Image.open(i)
#    im = im.transpose( Image.ROTATE_270 )
#    im.save(i)
def change(R,G,B):
	width, height = im.size
	for y in range(height):
		for x in range(width):
		    rgba = im.getpixel((x,y))
		    rgba = (rgba[0]-R,  # R
		            rgba[1]-G,  # G
		            rgba[2]-B  # B
		                  ); # A
		    im.putpixel((x,y), rgba)
#print(im.size)
def cut(location):
    for i in os.listdir('./image/'+location):
        im = Image.open('./image/'+location+'/'+i)
#    im = im.transpose( Image.ROTATE_180 )
        x = 600
        y = 2100
        w = 50
        h = 300
        im = im.crop((x, y, x+w, y+h))
#        plt.imshow(im)
#        plt.show()
        im.save('./test/'+location+'/'+i)


#plt.imshow(im)
#plt.show()
if __name__ == "__main__":
    process_count=0
    record=[]
    p=multiprocessing.Pool()

    for i in os.listdir('./image'):
        for j in os.listdir('./image/'+i):
            for k in os.listdir('./image/'+i+'/'+j):
                process_count+=1
                p = multiprocessing.Process(target=cut, args=(i+'/'+j+'/'+k,))
                p.start()
                record.append(p)
                print(process_count)
                if(process_count/3==1):
                    for process in record:
                        process.join()
                        process_count=0
                        record=[]





