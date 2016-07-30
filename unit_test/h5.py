import h5py
import Image
import time

f= h5py.File("../dataset/storytelling.h5","r")
data = f['images']

for i in range(data.shape[0]):
    if i%5 == 0:
        im = data[i][:][:][:]
        im = Image.fromarray(im.transpose(1,2,0))
        im.show()
        time.sleep(1)


print(1)