from models.video_base_model import VideoBaseModel
from utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class EDVRModel(VideoBaseModel):
    def __init__(self, opt):
        super(EDVRModel, self).__init__(opt)
        if self.is_train:
            self.train_tsa_iter = opt['train'].get('tsa_iter')

    def setup_optimizers(self):
        train_opt = self.opt['train']
        dcn_lr_mul = train_opt.get('dcn_lr_mul', 1)
        if dcn_lr_mul == 1:
            optim_params = self.net.parameters()
        else:
            normal_params = []
            dcn_params = []
            for name, param in self.net.named_parameters():
                if 'dcn' in name:
                    dcn_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [{'params': normal_params,
                             'lr': train_opt['optim']['lr']},
                            {'params': dcn_params,
                             'lr': train_opt['optim']['lr']*dcn_lr_mul}]

        optim_type = train_opt['optim'].pop('type')
        self.optimizer = self._get_optimizer(optim_type, optim_params,
                                            **train_opt['optim'])

    def optimize_parameters(self, current_iter):
        if self.train_tsa_iter:
            if current_iter == 1:
                for name, param in self.net.named_parameters():
                    if 'fusion' not in name:
                        param.requires_grad = False
            elif current_iter == self.train_isa_iter:
                for param in self.net.parameters():
                    param.requires_grad = True

        super(EDVRModel, self).optimize_parameters(current_iter)
