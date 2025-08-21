import math


class CosineDecay:
	def __init__(self,
	             optimizer,
	             max_lr,
	             min_lr,
	             max_epoch,
	             test_mode=False):
		self.optimizer = optimizer
		self.max_lr = max_lr
		self.min_lr = min_lr
		self.max_epoch = max_epoch
		self.test_mode = test_mode

		self.current_lr = max_lr
		self.cnt = 0
		if self.max_epoch > 1:
			self.scale = (max_lr - min_lr) / 2
			self.shift = (max_lr + min_lr) / 2
			self.alpha = math.pi / (max_epoch - 1)

	def step(self):
		self.cnt += 1
		self.current_lr = self.scale * math.cos(self.alpha * self.cnt) + self.shift

		if not self.test_mode:
			for param_group in self.optimizer.param_groups:
				param_group['lr'] = self.current_lr

	def get_lr(self):
		return self.current_lr

    # === Added for checkpointing ===
	def state_dict(self):
		return {
            "cnt": self.cnt,
            "current_lr": self.current_lr
        }
	
	def load_state_dict(self, state_dict):
		self.cnt = state_dict["cnt"]
		self.current_lr = state_dict["current_lr"]

        # restore lr to optimizer
		if not self.test_mode and self.optimizer is not None:
			for param_group in self.optimizer.param_groups:
				param_group['lr'] = self.current_lr

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	max_epoch = 100
	s = CosineDecay(None, max_lr=2.5e-4, min_lr=1e-7, max_epoch=max_epoch, test_mode=True)
	lr_list = []
	for i in range(max_epoch):
		lr_list.append(s.get_lr())
		s.step()

	plt.plot(list(range(max_epoch)), lr_list)
	plt.show()
