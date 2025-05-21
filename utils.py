def save_image_mat():
    mat = scipy.io.loadmat(DATA_PATH+"Semi-automatic-annotations/001.mat")
    image = mat['mask']
    print(f'Ground-truth size: {image.shape}')
    plt.imshow(image)
    plt.savefig("mat_mask.png")
    plt.close()

def debug_cuda():
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)