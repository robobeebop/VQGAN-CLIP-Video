from utils import *

class Dream:
    def __init__(self, isHighVRAM=True) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        self.resLimit = 4.2e5 if isHighVRAM else 2.5e5

    def cook(self, vqgan_path, cut_n=32, cut_pow=1., prompts="", init_weight=4, clip_model='ViT-B/16'):
        self.vqgan_config = vqgan_path[0]
        self.vqgan_checkpoint = vqgan_path[1]
        self.model = load_vqgan_model(self.vqgan_config, self.vqgan_checkpoint).to(self.device)

        self.clip_model = clip_model
        self.perceptor = clip.load(self.clip_model, jit=False)[0].eval().requires_grad_(False).to(self.device)

        self.cut_size = self.perceptor.visual.input_resolution
        self.e_dim = self.model.quantize.e_dim
        self.f = 2**(self.model.decoder.num_resolutions - 1)
        self.make_cutouts = MakeCutouts(self.cut_size, cutn=cut_n, cut_pow=cut_pow)
        
        self.z_min = self.model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        self.z_max = self.model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
        self.pMs = []
        self.init_weight = init_weight
        prompts = prompts.split("|")

        for prompt in prompts:
            txt, weight, stop = parse_prompt(prompt)
            embed = self.perceptor.encode_text(clip.tokenize(txt).to(self.device)).float()
            self.pMs.append(Prompt(embed, weight, stop).to(self.device))
    
        self.pMs.append(Prompt(embed, weight).to(self.device))
    
    def deepdream(self, init_image, iter_n=25, step_size=0.3):
        
        pil_image = Image.fromarray((init_image * 1).astype(np.uint8)).convert('RGB')
        self.z, *_ = self.model.encode(TF.to_tensor(pil_image).to(self.device).unsqueeze(0) * 2 - 1)

        self.z_orig = self.z.clone()
        self.z.requires_grad_(True)
        self.opt = optim.Adam([self.z], lr=step_size)


        gen = torch.Generator().manual_seed(0)
        embed = torch.empty([1, self.perceptor.visual.output_dim]).normal_(generator=gen)

        try:
            for i in range(iter_n):
                out = self.train(i, iter_n)
        except KeyboardInterrupt:
            pass

        return np.float32(TF.to_pil_image(out[0].cpu()))

    def train(self, i, iter_n):
        torch.set_grad_enabled(True)
        self.opt.zero_grad()
        lossAll = self.ascend_txt()
        loss = sum(lossAll)
        loss.backward()
        torch.set_grad_enabled(False)
        self.opt.step()

        with torch.no_grad():
            self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))

        if i == iter_n-1:
            return self.checkout()
        
        return None


    def ascend_txt(self):
        out = self.synth()
        iii = self.perceptor.encode_image(self.normalize(self.make_cutouts(out))).float()
        result = []
        if self.init_weight:
            result.append(F.mse_loss(self.z, self.z_orig) * self.init_weight / 2)
        for prompt in self.pMs:
            result.append(prompt(iii))
        return result

    def synth(self):
        z_q = vector_quantize(self.z.movedim(1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
        return clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)

    @torch.no_grad()
    def checkout(self):
        out = self.synth()
        return out