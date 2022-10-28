from rich.progress import track


def convert_images_to_1d(data_loader):
	if data_loader is None:
		return None, None
	X, y = [], []
	for images, targets in track(data_loader, description="Converting images..."):
		# average spatial dimension of images (only not nan values)
		signatures = images.nanmean((2, 3)).numpy()
		targets = targets.numpy()
		[X.append(sig) for sig in signatures]
		[y.append(tar) for tar in targets]
	return X, y




