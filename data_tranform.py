from torchvision.transforms.functional import to_tensor



class CropPatches(object):
    def __init__(self, patch_size=112, stride=80):
        self.patch_size = patch_size
        self.stride = stride

    def __call__(self, image):
        w, h = image.size

        patches = ()
        for i in range(0, h-self.stride, self.stride):
            for j in range(0, w-self.stride, self.stride):
                if j+self.patch_size > w or i+self.patch_size > h:
                    pass
                else:
                    temp = image.crop((j, i, j+self.patch_size, i+self.patch_size))
                    patch = to_tensor(temp)
                    patches = patches + (patch,)

        return patches




