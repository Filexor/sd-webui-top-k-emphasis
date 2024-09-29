from math import ceil
import torch
import einops

from modules.devices import device

from scripts.parsing import EmphasisPair

class Emphasis:
    name: str = "Base"
    description: str = ""
    tokens: list[list[int]]
    multipliers: list[list[EmphasisPair]]
    z: torch.Tensor # [batch, token, channel]

    def after_transformers(self):
        pass

class TopKEmphasis(Emphasis):
    name = "Top K Emphasis"
    description = "Expanded emphasis method"

    def __init__(self, emphasis_view_update=False, debug=False, embedding_key='clip_l') -> None:
        super().__init__()
        self.emphasis_view_update = emphasis_view_update
        self.debug = debug
        self.embedding_key = embedding_key

    def after_transformers_old(self):
        for i in range(self.z.shape[0]):
            for j in range(self.z.shape[1]):
                z_des = self.z[i, j, :].sort(descending=True).values
                for k in self.multipliers[i][j].multipliers:
                    if k.key == "c":
                        if k.threshold < 0.0:
                            raise ValueError("Negative threshold is unacceptable.")
                        elif k.threshold == 0.0:
                            self.z[i, j, :] *= k.weight
                        elif k.threshold < 1.0:
                            threshold = z_des[int(k.threshold * self.z.shape[2]) - 1]
                            self.z[i, j, :] *= torch.where(self.z[i, j, :] >= threshold, k.weight, 1.0)
                        else:
                            threshold = z_des[min(int(k.threshold), self.z.shape[2]) - 1]
                            self.z[i, j, :] *= torch.where(self.z[i, j, :] >= threshold, k.weight, 1.0)

    def after_transformers_old2(self):
        for i, pair in enumerate(self.multipliers):
            multiplier, threshold = to_structure_of_tensor(pair, "c")
            threshold = torch.where(threshold == 0.0, self.z.shape[2], threshold)
            threshold = torch.where(threshold < 1.0, threshold * self.z.shape[2], threshold)
            threshold = threshold - 1
            threshold = threshold.to(dtype=torch.int32).to(device)
            multiplier = multiplier.to(device)
            for j in range(threshold.shape[0]):
                z_des = self.z[i, ...].sort(dim=1, descending=True).values
                selected_z_des = z_des.index_select(dim=1, index=threshold[j, :]).diag()
                expanded_z_des = selected_z_des.unsqueeze(1).expand(-1, self.z.shape[2])
                expanded_multiplier = multiplier[j, :].unsqueeze(1).expand(-1, self.z.shape[2])
                self.z[i, ...] *= torch.where(self.z[i, ...] >= expanded_z_des, expanded_multiplier, 1.0)

    def after_transformers(self):
        if self.emphasis_view_update:
            for i, pairs in enumerate(self.multipliers):
                for pair in pairs:
                    for multiplier in pair.multipliers:
                        if multiplier.key == "c" or (multiplier.key == "l" and self.embedding_key == "clip_l") or (multiplier.key == "g" and self.embedding_key == "clip_g"):
                            mode = None
                            value = None
                            preoffset = 0.0
                            postoffset = 0.0
                            for option in multiplier.options:
                                match option[0]:
                                    case "b" | "o" | "m" | "r":
                                        if mode is None:
                                            mode = option[0]
                                            if option[1] is not None:
                                                value = option[1]
                                            else:
                                                value = multiplier.threshold
                                    case "c":
                                        if mode is None:
                                            mode = option[0]
                                            if option[1] is not None:
                                                value = option[1]
                                            else:
                                                value = multiplier.threshold + 1
                                    case "n":
                                        pass    # This is not cross attention.
                                        # if mode is None and option[1] is not None:
                                        #     mode = option[0]
                                        #     value = option[1]
                                    case "pa":
                                        if option[1] is not None:
                                            preoffset += option[1]
                                    case "ps":
                                        if option[1] is not None:
                                            preoffset -= option[1]
                                    case "a":
                                        if option[1] is not None:
                                            postoffset += option[1]
                                    case "s":
                                        if option[1] is not None:
                                            postoffset -= option[1]
                            weight = torch.asarray([multiplier.weight], dtype=torch.float32, device=self.z.device)
                            preoffset = torch.asarray([preoffset], dtype=torch.float32, device=self.z.device)
                            postoffset = torch.asarray([postoffset], dtype=torch.float32, device=self.z.device)
                            zero = torch.asarray([0.0], dtype=torch.float32, device=self.z.device)
                            one = torch.asarray([1.0], dtype=torch.float32, device=self.z.device)
                            match mode:
                                case None:
                                    thres_top = multiplier.threshold
                                    thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                    z_des = self.z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                    target = self.z[i, pair.begin:pair.end, :] >= z_des_exp
                                    if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                    self.z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    self.z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    self.z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                case "b":
                                    if multiplier.threshold == 0.0 and value == 0.0:
                                        if self.debug: print("Emphasis will be skipped.")
                                        pass
                                    elif multiplier.threshold != 0.0 and value == 0.0:
                                        thres_top = multiplier.threshold
                                        #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                        thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                        thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                        thres_top = thres_top - 1
                                        thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                        z_des = self.z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                        z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                        z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        target = self.z[i, pair.begin:pair.end, :] >= z_des_exp
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        self.z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                        self.z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                        self.z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                    elif multiplier.threshold == 0.0 and value != 0.0:
                                        thres_top = value
                                        #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                        thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                        thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                        thres_top = thres_top - 1
                                        thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                        z_des = self.z[i, pair.begin:pair.end, :].sort(dim=-1, descending=False).values
                                        z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                        z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        target = self.z[i, pair.begin:pair.end, :] <= z_des_exp
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        self.z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                        self.z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                        self.z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                    elif multiplier.threshold != 0.0 and value != 0.0:
                                        thres_top = multiplier.threshold
                                        #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                        thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                        thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                        thres_top = thres_top - 1
                                        thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                        thres_bottom = value
                                        #thres_bottom = self.z.shape[2] if thres_bottom == 0 else thres_bottom
                                        thres_bottom = thres_bottom * self.z.shape[2] if thres_bottom < 1 else thres_bottom
                                        thres_bottom = self.z.shape[2] if thres_bottom > self.z.shape[2] else thres_bottom
                                        thres_bottom = thres_bottom - 1
                                        thres_bottom = torch.asarray([thres_bottom]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                        z_des = self.z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                        z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                        z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        z_asc = z_des.flip([-1])
                                        z_asc_sel = z_asc.index_select(dim=-1, index=thres_bottom).diag()
                                        z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        target = (self.z[i, pair.begin:pair.end, :] >= z_des_exp) | (self.z[i, pair.begin:pair.end, :] <= z_asc_exp)
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        self.z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                        self.z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                        self.z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                case "o":
                                    if multiplier.threshold == 0.0 and value == 0.0:
                                        if self.debug: print("Emphasis will be applied to all elements in specified tokens.")
                                        self.z[i, pair.begin:pair.end, :] += preoffset
                                        self.z[i, pair.begin:pair.end, :] *= weight
                                        self.z[i, pair.begin:pair.end, :] += postoffset
                                    elif multiplier.threshold != 0.0 and value == 0.0:
                                        thres_top = multiplier.threshold
                                        #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                        thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                        thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                        thres_top = thres_top - 1
                                        thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                        z_des = self.z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                        z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                        z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        target = self.z[i, pair.begin:pair.end, :] < z_des_exp
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        self.z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                        self.z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                        self.z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                    elif multiplier.threshold == 0.0 and value != 0.0:
                                        thres_top = value
                                        #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                        thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                        thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                        thres_top = thres_top - 1
                                        thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                        z_des = self.z[i, pair.begin:pair.end, :].sort(dim=-1, descending=False).values
                                        z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                        z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        target = self.z[i, pair.begin:pair.end, :] > z_des_exp
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        self.z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                        self.z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                        self.z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                    elif multiplier.threshold != 0.0 and value != 0.0:
                                        thres_top = multiplier.threshold
                                        #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                        thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                        thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                        thres_top = thres_top - 1
                                        thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                        thres_bottom = value
                                        #thres_bottom = self.z.shape[2] if thres_bottom == 0 else thres_bottom
                                        thres_bottom = thres_bottom * self.z.shape[2] if thres_bottom < 1 else thres_bottom
                                        thres_bottom = self.z.shape[2] if thres_bottom > self.z.shape[2] else thres_bottom
                                        thres_bottom = thres_bottom - 1
                                        thres_bottom = torch.asarray([thres_bottom]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                        z_des = self.z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                        z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                        z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        z_asc = z_des.flip([-1])
                                        z_asc_sel = z_asc.index_select(dim=-1, index=thres_bottom).diag()
                                        z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        target = (self.z[i, pair.begin:pair.end, :] < z_des_exp) & (self.z[i, pair.begin:pair.end, :] > z_asc_exp)
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        self.z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                        self.z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                        self.z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                case "m":
                                    thres_top = multiplier.threshold
                                    thres_top = (self.z.shape[2] // 2) - thres_top if thres_top >= 1 else thres_top
                                    thres_top = (self.z.shape[2] // 2) - thres_top * (self.z.shape[2] // 2) if thres_top < 1 else thres_top
                                    thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                    thres_bottom = value
                                    thres_bottom = (self.z.shape[2] // 2) - thres_bottom if thres_bottom >= 1 else thres_bottom
                                    thres_bottom = (self.z.shape[2] // 2) - thres_bottom * (self.z.shape[2] // 2) if thres_bottom < 1 else thres_bottom
                                    thres_bottom = self.z.shape[2] if thres_bottom > self.z.shape[2] else thres_bottom
                                    thres_bottom = thres_bottom - 1
                                    thres_bottom = torch.asarray([thres_bottom]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                    z_des = self.z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                    z_asc = z_des.flip([-1])
                                    z_asc_sel = z_asc.index_select(dim=-1, index=thres_bottom).diag()
                                    z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                    target = (self.z[i, pair.begin:pair.end, :] < z_des_exp) & (self.z[i, pair.begin:pair.end, :] > z_asc_exp)
                                    if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                    self.z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    self.z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    self.z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                case "r":
                                    thres_top = multiplier.threshold
                                    #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = self.z.shape[2] - 1 if thres_top > self.z.shape[2] - 1 else thres_top
                                    #thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                    thres_bottom = value
                                    #thres_bottom = self.z.shape[2] if thres_bottom == 0 else thres_bottom
                                    thres_bottom = thres_bottom * self.z.shape[2] if thres_bottom < 1 else thres_bottom
                                    thres_bottom = self.z.shape[2] - 1 if thres_bottom > self.z.shape[2] - 1 else thres_bottom
                                    #thres_bottom = thres_bottom - 1
                                    thres_bottom = torch.asarray([thres_bottom]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                    z_des = self.z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                    z_asc = z_des
                                    z_asc_sel = z_asc.index_select(dim=-1, index=thres_bottom).diag()
                                    z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                    target = (self.z[i, pair.begin:pair.end, :] <= z_des_exp) & (self.z[i, pair.begin:pair.end, :] >= z_asc_exp)
                                    if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                    self.z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    self.z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    self.z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                case "c":
                                    thres_top = multiplier.threshold
                                    #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = self.z.shape[2] - 1 if thres_top > self.z.shape[2] else thres_top
                                    thres_top = int(thres_top)
                                    #thres_top = thres_top - 1
                                    thres_bottom = value
                                    #thres_bottom = self.z.shape[2] if thres_bottom == 0 else thres_bottom
                                    thres_bottom = thres_bottom * self.z.shape[2] if thres_bottom < 1 else thres_bottom
                                    thres_bottom = self.z.shape[2] if thres_bottom > self.z.shape[2] else thres_bottom
                                    # thres_bottom = thres_bottom - 1
                                    thres_bottom = int(thres_bottom)
                                    self.z[i, pair.begin:pair.end, thres_top:thres_bottom] += preoffset
                                    self.z[i, pair.begin:pair.end, thres_top:thres_bottom] *= weight
                                    self.z[i, pair.begin:pair.end, thres_top:thres_bottom] += postoffset
        else:
            for i, pairs in enumerate(self.multipliers):
                for pair in pairs:
                    z_des = self.z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                    z_asc = z_des.flip([-1])
                    for multiplier in pair.multipliers:
                        if multiplier.key == "c" or (multiplier.key == "l" and self.embedding_key == "clip_l") or (multiplier.key == "g" and self.embedding_key == "clip_g"):
                            mode = None
                            value = None
                            preoffset = 0.0
                            postoffset = 0.0
                            for option in multiplier.options:
                                match option[0]:
                                    case "b" | "o" | "m" | "r":
                                        if mode is None:
                                            mode = option[0]
                                            if option[1] is not None:
                                                value = option[1]
                                            else:
                                                value = multiplier.threshold
                                    case "c":
                                        if mode is None:
                                            mode = option[0]
                                            if option[1] is not None:
                                                value = option[1]
                                            else:
                                                value = multiplier.threshold + 1
                                    case "n":
                                        pass    # This is not cross attention.
                                        # if mode is None and option[1] is not None:
                                        #     mode = option[0]
                                        #     value = option[1]
                                    case "pa":
                                        if option[1] is not None:
                                            preoffset += option[1]
                                    case "ps":
                                        if option[1] is not None:
                                            preoffset -= option[1]
                                    case "a":
                                        if option[1] is not None:
                                            postoffset += option[1]
                                    case "s":
                                        if option[1] is not None:
                                            postoffset -= option[1]
                            weight = torch.asarray([multiplier.weight], dtype=torch.float32, device=self.z.device)
                            preoffset = torch.asarray([preoffset], dtype=torch.float32, device=self.z.device)
                            postoffset = torch.asarray([postoffset], dtype=torch.float32, device=self.z.device)
                            zero = torch.asarray([0.0], dtype=torch.float32, device=self.z.device)
                            one = torch.asarray([1.0], dtype=torch.float32, device=self.z.device)
                            match mode:
                                case None:
                                    thres_top = multiplier.threshold
                                    thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                    #z_des = self.z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                    target = self.z[i, pair.begin:pair.end, :] >= z_des_exp
                                    if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                    self.z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    self.z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    self.z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                case "b":
                                    if multiplier.threshold == 0.0 and value == 0.0:
                                        if self.debug: print("Emphasis will be skipped.")
                                        pass
                                    elif multiplier.threshold != 0.0 and value == 0.0:
                                        thres_top = multiplier.threshold
                                        #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                        thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                        thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                        thres_top = thres_top - 1
                                        thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                        #z_des = self.z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                        z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                        z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        target = self.z[i, pair.begin:pair.end, :] >= z_des_exp
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        self.z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                        self.z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                        self.z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                    elif multiplier.threshold == 0.0 and value != 0.0:
                                        thres_top = value
                                        #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                        thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                        thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                        thres_top = thres_top - 1
                                        thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                        #z_des = self.z[i, pair.begin:pair.end, :].sort(dim=-1, descending=False).values
                                        z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                        z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        target = self.z[i, pair.begin:pair.end, :] <= z_des_exp
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        self.z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                        self.z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                        self.z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                    elif multiplier.threshold != 0.0 and value != 0.0:
                                        thres_top = multiplier.threshold
                                        #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                        thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                        thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                        thres_top = thres_top - 1
                                        thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                        thres_bottom = value
                                        #thres_bottom = self.z.shape[2] if thres_bottom == 0 else thres_bottom
                                        thres_bottom = thres_bottom * self.z.shape[2] if thres_bottom < 1 else thres_bottom
                                        thres_bottom = self.z.shape[2] if thres_bottom > self.z.shape[2] else thres_bottom
                                        thres_bottom = thres_bottom - 1
                                        thres_bottom = torch.asarray([thres_bottom]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                        #z_des = self.z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                        z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                        z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        #z_asc = z_des.flip([-1])
                                        z_asc_sel = z_asc.index_select(dim=-1, index=thres_bottom).diag()
                                        z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        target = (self.z[i, pair.begin:pair.end, :] >= z_des_exp) | (self.z[i, pair.begin:pair.end, :] <= z_asc_exp)
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        self.z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                        self.z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                        self.z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                case "o":
                                    if multiplier.threshold == 0.0 and value == 0.0:
                                        if self.debug: print("Emphasis will be applied to all elements in specified tokens.")
                                        self.z[i, pair.begin:pair.end, :] += preoffset
                                        self.z[i, pair.begin:pair.end, :] *= weight
                                        self.z[i, pair.begin:pair.end, :] += postoffset
                                    elif multiplier.threshold != 0.0 and value == 0.0:
                                        thres_top = multiplier.threshold
                                        #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                        thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                        thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                        thres_top = thres_top - 1
                                        thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                        #z_des = self.z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                        z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                        z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        target = self.z[i, pair.begin:pair.end, :] < z_des_exp
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        self.z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                        self.z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                        self.z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                    elif multiplier.threshold == 0.0 and value != 0.0:
                                        thres_top = value
                                        #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                        thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                        thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                        thres_top = thres_top - 1
                                        thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                        #z_des = self.z[i, pair.begin:pair.end, :].sort(dim=-1, descending=False).values
                                        z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                        z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        target = self.z[i, pair.begin:pair.end, :] > z_des_exp
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        self.z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                        self.z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                        self.z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                    elif multiplier.threshold != 0.0 and value != 0.0:
                                        thres_top = multiplier.threshold
                                        #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                        thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                        thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                        thres_top = thres_top - 1
                                        thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                        thres_bottom = value
                                        #thres_bottom = self.z.shape[2] if thres_bottom == 0 else thres_bottom
                                        thres_bottom = thres_bottom * self.z.shape[2] if thres_bottom < 1 else thres_bottom
                                        thres_bottom = self.z.shape[2] if thres_bottom > self.z.shape[2] else thres_bottom
                                        thres_bottom = thres_bottom - 1
                                        thres_bottom = torch.asarray([thres_bottom]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                        #z_des = self.z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                        z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                        z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        #z_asc = z_des.flip([-1])
                                        z_asc_sel = z_asc.index_select(dim=-1, index=thres_bottom).diag()
                                        z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                        target = (self.z[i, pair.begin:pair.end, :] < z_des_exp) & (self.z[i, pair.begin:pair.end, :] > z_asc_exp)
                                        if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                        self.z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                        self.z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                        self.z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                case "m":
                                    thres_top = multiplier.threshold
                                    thres_top = (self.z.shape[2] // 2) - thres_top if thres_top >= 1 else thres_top
                                    thres_top = (self.z.shape[2] // 2) - thres_top * (self.z.shape[2] // 2) if thres_top < 1 else thres_top
                                    thres_top = self.z.shape[2] if thres_top > self.z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                    thres_bottom = value
                                    thres_bottom = (self.z.shape[2] // 2) - thres_bottom if thres_bottom >= 1 else thres_bottom
                                    thres_bottom = (self.z.shape[2] // 2) - thres_bottom * (self.z.shape[2] // 2) if thres_bottom < 1 else thres_bottom
                                    thres_bottom = self.z.shape[2] if thres_bottom > self.z.shape[2] else thres_bottom
                                    thres_bottom = thres_bottom - 1
                                    thres_bottom = torch.asarray([thres_bottom]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                    #z_des = self.z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                    #z_asc = z_des.flip([-1])
                                    z_asc_sel = z_asc.index_select(dim=-1, index=thres_bottom).diag()
                                    z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                    target = (self.z[i, pair.begin:pair.end, :] < z_des_exp) & (self.z[i, pair.begin:pair.end, :] > z_asc_exp)
                                    if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                    self.z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    self.z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    self.z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                case "r":
                                    thres_top = multiplier.threshold
                                    #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = self.z.shape[2] - 1 if thres_top > self.z.shape[2] - 1 else thres_top
                                    #thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                    thres_bottom = value
                                    #thres_bottom = self.z.shape[2] if thres_bottom == 0 else thres_bottom
                                    thres_bottom = thres_bottom * self.z.shape[2] if thres_bottom < 1 else thres_bottom
                                    thres_bottom = self.z.shape[2] - 1 if thres_bottom > self.z.shape[2] - 1 else thres_bottom
                                    #thres_bottom = thres_bottom - 1
                                    thres_bottom = torch.asarray([thres_bottom]).to(dtype=torch.int32, device=self.z.device).expand((pair.end-pair.begin, ))
                                    #z_des = self.z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                    #z_asc = z_des
                                    z_asc_sel = z_des.index_select(dim=-1, index=thres_bottom).diag()
                                    z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, self.z.shape[2]))
                                    target = (self.z[i, pair.begin:pair.end, :] <= z_des_exp) & (self.z[i, pair.begin:pair.end, :] >= z_asc_exp)
                                    if self.debug: print(target.to(device="cpu").nonzero().tolist())
                                    self.z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    self.z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    self.z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                case "c":
                                    thres_top = multiplier.threshold
                                    #thres_top = self.z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * self.z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = self.z.shape[2] - 1 if thres_top > self.z.shape[2] else thres_top
                                    thres_top = int(thres_top)
                                    #thres_top = thres_top - 1
                                    thres_bottom = value
                                    #thres_bottom = self.z.shape[2] if thres_bottom == 0 else thres_bottom
                                    thres_bottom = thres_bottom * self.z.shape[2] if thres_bottom < 1 else thres_bottom
                                    thres_bottom = self.z.shape[2] if thres_bottom > self.z.shape[2] else thres_bottom
                                    # thres_bottom = thres_bottom - 1
                                    thres_bottom = int(thres_bottom)
                                    self.z[i, pair.begin:pair.end, thres_top:thres_bottom] += preoffset
                                    self.z[i, pair.begin:pair.end, thres_top:thres_bottom] *= weight
                                    self.z[i, pair.begin:pair.end, thres_top:thres_bottom] += postoffset

def to_structure_of_tensor(input: list[EmphasisPair], key: str) -> tuple[torch.Tensor, torch.Tensor]: 
    count = 0
    weights = []
    thresholds = []
    for i in input:
        weight = []
        threshold = []
        count_tokenwise = 0
        for j in i.multipliers:
            if j.key == key:
                count_tokenwise += 1
                weight.append(j.weight)
                threshold.append(j.threshold)
        count = max(count, count_tokenwise)
        weights.append(weight)
        thresholds.append(threshold)
    for i in weights:
        i += [1.0] * (count - len(i))
    for i in thresholds:
        i += [0.0] * (count - len(i))
    weight = torch.asarray(weights)
    weight = einops.rearrange(weight, "a b -> b a")
    threshold = torch.asarray(thresholds)
    threshold = einops.rearrange(threshold, "a b -> b a")
    return weight, threshold

def emphasis_b(z, multipliers, emphasis_view_update, embedding_key, debug):
    if emphasis_view_update:
        for i, pairs in enumerate(multipliers):
            for pair in pairs:
                for multiplier in pair.multipliers:
                    if multiplier.key == "b" or (multiplier.key == "bl" and embedding_key == "clip_l") or (multiplier.key == "bg" and embedding_key == "clip_g"):
                        mode = None
                        value = None
                        preoffset = 0.0
                        postoffset = 0.0
                        for option in multiplier.options:
                            match option[0]:
                                case "b" | "o" | "m" | "r":
                                    if mode is None:
                                        mode = option[0]
                                        if option[1] is not None:
                                            value = option[1]
                                        else:
                                            value = multiplier.threshold
                                case "c":
                                    if mode is None:
                                        mode = option[0]
                                        if option[1] is not None:
                                            value = option[1]
                                        else:
                                            value = multiplier.threshold + 1
                                case "n":
                                    pass    # This is not cross attention.
                                    # if mode is None and option[1] is not None:
                                    #     mode = option[0]
                                    #     value = option[1]
                                case "pa":
                                    if option[1] is not None:
                                        preoffset += option[1]
                                case "ps":
                                    if option[1] is not None:
                                        preoffset -= option[1]
                                case "a":
                                    if option[1] is not None:
                                        postoffset += option[1]
                                case "s":
                                    if option[1] is not None:
                                        postoffset -= option[1]
                        weight = torch.asarray([multiplier.weight], dtype=torch.float32, device=z.device)
                        preoffset = torch.asarray([preoffset], dtype=torch.float32, device=z.device)
                        postoffset = torch.asarray([postoffset], dtype=torch.float32, device=z.device)
                        zero = torch.asarray([0.0], dtype=torch.float32, device=z.device)
                        one = torch.asarray([1.0], dtype=torch.float32, device=z.device)
                        match mode:
                            case None:
                                thres_top = multiplier.threshold
                                thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = z[i, pair.begin:pair.end, :] >= z_des_exp
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                            case "b":
                                if multiplier.threshold == 0.0 and value == 0.0:
                                    if debug: print("Emphasis will be skipped.")
                                    pass
                                elif multiplier.threshold != 0.0 and value == 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, pair.begin:pair.end, :] >= z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                elif multiplier.threshold == 0.0 and value != 0.0:
                                    thres_top = value
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=False).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, pair.begin:pair.end, :] <= z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                elif multiplier.threshold != 0.0 and value != 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    thres_bottom = value
                                    #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                    thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                    thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                    thres_bottom = thres_bottom - 1
                                    thres_bottom = torch.asarray([thres_bottom]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    z_asc = z_des.flip([-1])
                                    z_asc_sel = z_asc.index_select(dim=-1, index=thres_bottom).diag()
                                    z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = (z[i, pair.begin:pair.end, :] >= z_des_exp) | (z[i, pair.begin:pair.end, :] <= z_asc_exp)
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                            case "o":
                                if multiplier.threshold == 0.0 and value == 0.0:
                                    if debug: print("Emphasis will be applied to all elements in specified tokens.")
                                    z[i, pair.begin:pair.end, :] += preoffset
                                    z[i, pair.begin:pair.end, :] *= weight
                                    z[i, pair.begin:pair.end, :] += postoffset
                                elif multiplier.threshold != 0.0 and value == 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, pair.begin:pair.end, :] < z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                elif multiplier.threshold == 0.0 and value != 0.0:
                                    thres_top = value
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=False).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, pair.begin:pair.end, :] > z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                elif multiplier.threshold != 0.0 and value != 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    thres_bottom = value
                                    #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                    thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                    thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                    thres_bottom = thres_bottom - 1
                                    thres_bottom = torch.asarray([thres_bottom]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    z_asc = z_des.flip([-1])
                                    z_asc_sel = z_asc.index_select(dim=-1, index=thres_bottom).diag()
                                    z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = (z[i, pair.begin:pair.end, :] < z_des_exp) & (z[i, pair.begin:pair.end, :] > z_asc_exp)
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                            case "m":
                                thres_top = multiplier.threshold
                                thres_top = (z.shape[2] // 2) - thres_top if thres_top >= 1 else thres_top
                                thres_top = (z.shape[2] // 2) - thres_top * (z.shape[2] // 2) if thres_top < 1 else thres_top
                                thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                thres_bottom = value
                                thres_bottom = (z.shape[2] // 2) - thres_bottom if thres_bottom >= 1 else thres_bottom
                                thres_bottom = (z.shape[2] // 2) - thres_bottom * (z.shape[2] // 2) if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                thres_bottom = thres_bottom - 1
                                thres_bottom = torch.asarray([thres_bottom]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                z_asc = z_des.flip([-1])
                                z_asc_sel = z_asc.index_select(dim=-1, index=thres_bottom).diag()
                                z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = (z[i, pair.begin:pair.end, :] < z_des_exp) & (z[i, pair.begin:pair.end, :] > z_asc_exp)
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                            case "r":
                                thres_top = multiplier.threshold
                                #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] - 1 if thres_top > z.shape[2] - 1 else thres_top
                                #thres_top = thres_top - 1
                                thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                thres_bottom = value
                                #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] - 1 if thres_bottom > z.shape[2] - 1 else thres_bottom
                                #thres_bottom = thres_bottom - 1
                                thres_bottom = torch.asarray([thres_bottom]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                z_asc = z_des
                                z_asc_sel = z_asc.index_select(dim=-1, index=thres_bottom).diag()
                                z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = (z[i, pair.begin:pair.end, :] <= z_des_exp) & (z[i, pair.begin:pair.end, :] >= z_asc_exp)
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                            case "c":
                                thres_top = multiplier.threshold
                                #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] - 1 if thres_top > z.shape[2] else thres_top
                                thres_top = int(thres_top)
                                #thres_top = thres_top - 1
                                thres_bottom = value
                                #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                # thres_bottom = thres_bottom - 1
                                thres_bottom = int(thres_bottom)
                                z[i, pair.begin:pair.end, thres_top:thres_bottom] += preoffset
                                z[i, pair.begin:pair.end, thres_top:thres_bottom] *= weight
                                z[i, pair.begin:pair.end, thres_top:thres_bottom] += postoffset
    else:
        for i, pairs in enumerate(multipliers):
            for pair in pairs:
                z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                z_asc = z_des.flip([-1])
                for multiplier in pair.multipliers:
                    if multiplier.key == "b" or (multiplier.key == "bl" and embedding_key == "clip_l") or (multiplier.key == "bg" and embedding_key == "clip_g"):
                        mode = None
                        value = None
                        preoffset = 0.0
                        postoffset = 0.0
                        for option in multiplier.options:
                            match option[0]:
                                case "b" | "o" | "m" | "r":
                                    if mode is None:
                                        mode = option[0]
                                        if option[1] is not None:
                                            value = option[1]
                                        else:
                                            value = multiplier.threshold
                                case "c":
                                    if mode is None:
                                        mode = option[0]
                                        if option[1] is not None:
                                            value = option[1]
                                        else:
                                            value = multiplier.threshold + 1
                                case "n":
                                    pass    # This is not cross attention.
                                    # if mode is None and option[1] is not None:
                                    #     mode = option[0]
                                    #     value = option[1]
                                case "pa":
                                    if option[1] is not None:
                                        preoffset += option[1]
                                case "ps":
                                    if option[1] is not None:
                                        preoffset -= option[1]
                                case "a":
                                    if option[1] is not None:
                                        postoffset += option[1]
                                case "s":
                                    if option[1] is not None:
                                        postoffset -= option[1]
                        weight = torch.asarray([multiplier.weight], dtype=torch.float32, device=z.device)
                        preoffset = torch.asarray([preoffset], dtype=torch.float32, device=z.device)
                        postoffset = torch.asarray([postoffset], dtype=torch.float32, device=z.device)
                        zero = torch.asarray([0.0], dtype=torch.float32, device=z.device)
                        one = torch.asarray([1.0], dtype=torch.float32, device=z.device)
                        match mode:
                            case None:
                                thres_top = multiplier.threshold
                                thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                #z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = z[i, pair.begin:pair.end, :] >= z_des_exp
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                            case "b":
                                if multiplier.threshold == 0.0 and value == 0.0:
                                    if debug: print("Emphasis will be skipped.")
                                    pass
                                elif multiplier.threshold != 0.0 and value == 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    #z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, pair.begin:pair.end, :] >= z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                elif multiplier.threshold == 0.0 and value != 0.0:
                                    thres_top = value
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    #z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=False).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, pair.begin:pair.end, :] <= z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                elif multiplier.threshold != 0.0 and value != 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    thres_bottom = value
                                    #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                    thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                    thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                    thres_bottom = thres_bottom - 1
                                    thres_bottom = torch.asarray([thres_bottom]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    #z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    #z_asc = z_des.flip([-1])
                                    z_asc_sel = z_asc.index_select(dim=-1, index=thres_bottom).diag()
                                    z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = (z[i, pair.begin:pair.end, :] >= z_des_exp) | (z[i, pair.begin:pair.end, :] <= z_asc_exp)
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                            case "o":
                                if multiplier.threshold == 0.0 and value == 0.0:
                                    if debug: print("Emphasis will be applied to all elements in specified tokens.")
                                    z[i, pair.begin:pair.end, :] += preoffset
                                    z[i, pair.begin:pair.end, :] *= weight
                                    z[i, pair.begin:pair.end, :] += postoffset
                                elif multiplier.threshold != 0.0 and value == 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    #z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, pair.begin:pair.end, :] < z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                elif multiplier.threshold == 0.0 and value != 0.0:
                                    thres_top = value
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    #z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=False).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, pair.begin:pair.end, :] > z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                elif multiplier.threshold != 0.0 and value != 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    thres_bottom = value
                                    #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                    thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                    thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                    thres_bottom = thres_bottom - 1
                                    thres_bottom = torch.asarray([thres_bottom]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    #z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    #z_asc = z_des.flip([-1])
                                    z_asc_sel = z_asc.index_select(dim=-1, index=thres_bottom).diag()
                                    z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = (z[i, pair.begin:pair.end, :] < z_des_exp) & (z[i, pair.begin:pair.end, :] > z_asc_exp)
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                            case "m":
                                thres_top = multiplier.threshold
                                thres_top = (z.shape[2] // 2) - thres_top if thres_top >= 1 else thres_top
                                thres_top = (z.shape[2] // 2) - thres_top * (z.shape[2] // 2) if thres_top < 1 else thres_top
                                thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                thres_bottom = value
                                thres_bottom = (z.shape[2] // 2) - thres_bottom if thres_bottom >= 1 else thres_bottom
                                thres_bottom = (z.shape[2] // 2) - thres_bottom * (z.shape[2] // 2) if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                thres_bottom = thres_bottom - 1
                                thres_bottom = torch.asarray([thres_bottom]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                #z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                #z_asc = z_des.flip([-1])
                                z_asc_sel = z_asc.index_select(dim=-1, index=thres_bottom).diag()
                                z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = (z[i, pair.begin:pair.end, :] < z_des_exp) & (z[i, pair.begin:pair.end, :] > z_asc_exp)
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                            case "r":
                                thres_top = multiplier.threshold
                                #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] - 1 if thres_top > z.shape[2] - 1 else thres_top
                                #thres_top = thres_top - 1
                                thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                thres_bottom = value
                                #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] - 1 if thres_bottom > z.shape[2] - 1 else thres_bottom
                                #thres_bottom = thres_bottom - 1
                                thres_bottom = torch.asarray([thres_bottom]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                #z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                #z_asc = z_des
                                z_asc_sel = z_des.index_select(dim=-1, index=thres_bottom).diag()
                                z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = (z[i, pair.begin:pair.end, :] <= z_des_exp) & (z[i, pair.begin:pair.end, :] >= z_asc_exp)
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                            case "c":
                                thres_top = multiplier.threshold
                                #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] - 1 if thres_top > z.shape[2] else thres_top
                                thres_top = int(thres_top)
                                #thres_top = thres_top - 1
                                thres_bottom = value
                                #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                # thres_bottom = thres_bottom - 1
                                thres_bottom = int(thres_bottom)
                                z[i, pair.begin:pair.end, thres_top:thres_bottom] += preoffset
                                z[i, pair.begin:pair.end, thres_top:thres_bottom] *= weight
                                z[i, pair.begin:pair.end, thres_top:thres_bottom] += postoffset
    return z

def emphasis_crossattention(z: torch.Tensor, multipliers_pos: list[list[EmphasisPair]], multipliers_neg: list[list[EmphasisPair]], key: str, crossattentioncounter, emphasis_view_update, debug):
    multipliers = multipliers_neg + multipliers_pos
    if emphasis_view_update:
        for i, pairs in enumerate(multipliers):
            for pair in pairs:
                for multiplier in pair.multipliers:
                    if multiplier.key == key:
                        mode = None
                        value = None
                        preoffset = 0.0
                        postoffset = 0.0
                        crossattentioncountertargets = []
                        for option in multiplier.options:
                            match option[0]:
                                case "b" | "o" | "m" | "r":
                                    if mode is None:
                                        mode = option[0]
                                        if option[1] is not None:
                                            value = option[1]
                                        else:
                                            value = multiplier.threshold
                                case "c":
                                    if mode is None:
                                        mode = option[0]
                                        if option[1] is not None:
                                            value = option[1]
                                        else:
                                            value = multiplier.threshold + 1
                                case "n":
                                    if option[1] is not None:
                                        crossattentioncountertargets.append(int(option[1]))
                                case "pa":
                                    if option[1] is not None:
                                        preoffset += option[1]
                                case "ps":
                                    if option[1] is not None:
                                        preoffset -= option[1]
                                case "a":
                                    if option[1] is not None:
                                        postoffset += option[1]
                                case "s":
                                    if option[1] is not None:
                                        postoffset -= option[1]
                        if len(crossattentioncountertargets) != 0 and crossattentioncounter not in crossattentioncountertargets:
                            break
                        weight = torch.asarray([multiplier.weight], dtype=torch.float32, device=z.device)
                        preoffset = torch.asarray([preoffset], dtype=torch.float32, device=z.device)
                        postoffset = torch.asarray([postoffset], dtype=torch.float32, device=z.device)
                        zero = torch.asarray([0.0], dtype=torch.float32, device=z.device)
                        one = torch.asarray([1.0], dtype=torch.float32, device=z.device)
                        match mode:
                            case None:
                                thres_top = multiplier.threshold
                                thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = z[i, pair.begin:pair.end, :] >= z_des_exp
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                            case "b":
                                if multiplier.threshold == 0.0 and value == 0.0:
                                    if debug: print("Emphasis will be skipped.")
                                    pass
                                elif multiplier.threshold != 0.0 and value == 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, pair.begin:pair.end, :] >= z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                elif multiplier.threshold == 0.0 and value != 0.0:
                                    thres_top = value
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=False).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, pair.begin:pair.end, :] <= z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                elif multiplier.threshold != 0.0 and value != 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    thres_bottom = value
                                    #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                    thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                    thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                    thres_bottom = thres_bottom - 1
                                    thres_bottom = torch.asarray([thres_bottom]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    z_asc = z_des.flip([-1])
                                    z_asc_sel = z_asc.index_select(dim=-1, index=thres_bottom).diag()
                                    z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = (z[i, pair.begin:pair.end, :] >= z_des_exp) | (z[i, pair.begin:pair.end, :] <= z_asc_exp)
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                            case "o":
                                if multiplier.threshold == 0.0 and value == 0.0:
                                    if debug: print("Emphasis will be applied to all elements in specified tokens.")
                                    z[i, pair.begin:pair.end, :] += preoffset
                                    z[i, pair.begin:pair.end, :] *= weight
                                    z[i, pair.begin:pair.end, :] += postoffset
                                elif multiplier.threshold != 0.0 and value == 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, pair.begin:pair.end, :] < z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                elif multiplier.threshold == 0.0 and value != 0.0:
                                    thres_top = value
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=False).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, pair.begin:pair.end, :] > z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                elif multiplier.threshold != 0.0 and value != 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    thres_bottom = value
                                    #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                    thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                    thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                    thres_bottom = thres_bottom - 1
                                    thres_bottom = torch.asarray([thres_bottom]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    z_asc = z_des.flip([-1])
                                    z_asc_sel = z_asc.index_select(dim=-1, index=thres_bottom).diag()
                                    z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = (z[i, pair.begin:pair.end, :] < z_des_exp) & (z[i, pair.begin:pair.end, :] > z_asc_exp)
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                            case "m":
                                thres_top = multiplier.threshold
                                thres_top = (z.shape[2] // 2) - thres_top if thres_top >= 1 else thres_top
                                thres_top = (z.shape[2] // 2) - thres_top * (z.shape[2] // 2) if thres_top < 1 else thres_top
                                thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                thres_bottom = value
                                thres_bottom = (z.shape[2] // 2) - thres_bottom if thres_bottom >= 1 else thres_bottom
                                thres_bottom = (z.shape[2] // 2) - thres_bottom * (z.shape[2] // 2) if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                thres_bottom = thres_bottom - 1
                                thres_bottom = torch.asarray([thres_bottom]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                z_asc = z_des.flip([-1])
                                z_asc_sel = z_asc.index_select(dim=-1, index=thres_bottom).diag()
                                z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = (z[i, pair.begin:pair.end, :] < z_des_exp) & (z[i, pair.begin:pair.end, :] > z_asc_exp)
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                            case "r":
                                thres_top = multiplier.threshold
                                #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] - 1 if thres_top > z.shape[2] - 1 else thres_top
                                #thres_top = thres_top - 1
                                thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                thres_bottom = value
                                #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] - 1 if thres_bottom > z.shape[2] - 1 else thres_bottom
                                #thres_bottom = thres_bottom - 1
                                thres_bottom = torch.asarray([thres_bottom]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                z_asc = z_des
                                z_asc_sel = z_asc.index_select(dim=-1, index=thres_bottom).diag()
                                z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = (z[i, pair.begin:pair.end, :] <= z_des_exp) & (z[i, pair.begin:pair.end, :] >= z_asc_exp)
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                            case "c":
                                thres_top = multiplier.threshold
                                #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] - 1 if thres_top > z.shape[2] else thres_top
                                thres_top = int(thres_top)
                                #thres_top = thres_top - 1
                                thres_bottom = value
                                #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                # thres_bottom = thres_bottom - 1
                                thres_bottom = int(thres_bottom)
                                z[i, pair.begin:pair.end, thres_top:thres_bottom] += preoffset
                                z[i, pair.begin:pair.end, thres_top:thres_bottom] *= weight
                                z[i, pair.begin:pair.end, thres_top:thres_bottom] += postoffset
    else:
        for i, pairs in enumerate(multipliers):
            for pair in pairs:
                z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                z_asc = z_des.flip([-1])
                for multiplier in pair.multipliers:
                    if multiplier.key == key:
                        mode = None
                        value = None
                        preoffset = 0.0
                        postoffset = 0.0
                        crossattentioncountertargets = []
                        for option in multiplier.options:
                            match option[0]:
                                case "b" | "o" | "m" | "r":
                                    if mode is None:
                                        mode = option[0]
                                        if option[1] is not None:
                                            value = option[1]
                                        else:
                                            value = multiplier.threshold
                                case "c":
                                    if mode is None:
                                        mode = option[0]
                                        if option[1] is not None:
                                            value = option[1]
                                        else:
                                            value = multiplier.threshold + 1
                                case "n":
                                    if option[1] is not None:
                                        crossattentioncountertargets.append(int(option[1]))
                                case "pa":
                                    if option[1] is not None:
                                        preoffset += option[1]
                                case "ps":
                                    if option[1] is not None:
                                        preoffset -= option[1]
                                case "a":
                                    if option[1] is not None:
                                        postoffset += option[1]
                                case "s":
                                    if option[1] is not None:
                                        postoffset -= option[1]
                        if len(crossattentioncountertargets) != 0 and crossattentioncounter not in crossattentioncountertargets:
                            break
                        weight = torch.asarray([multiplier.weight], dtype=torch.float32, device=z.device)
                        preoffset = torch.asarray([preoffset], dtype=torch.float32, device=z.device)
                        postoffset = torch.asarray([postoffset], dtype=torch.float32, device=z.device)
                        zero = torch.asarray([0.0], dtype=torch.float32, device=z.device)
                        one = torch.asarray([1.0], dtype=torch.float32, device=z.device)
                        match mode:
                            case None:
                                thres_top = multiplier.threshold
                                thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                #z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = z[i, pair.begin:pair.end, :] >= z_des_exp
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                            case "b":
                                if multiplier.threshold == 0.0 and value == 0.0:
                                    if debug: print("Emphasis will be skipped.")
                                    pass
                                elif multiplier.threshold != 0.0 and value == 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    #z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, pair.begin:pair.end, :] >= z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                elif multiplier.threshold == 0.0 and value != 0.0:
                                    thres_top = value
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    #z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=False).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, pair.begin:pair.end, :] <= z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                elif multiplier.threshold != 0.0 and value != 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    thres_bottom = value
                                    #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                    thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                    thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                    thres_bottom = thres_bottom - 1
                                    thres_bottom = torch.asarray([thres_bottom]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    #z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    #z_asc = z_des.flip([-1])
                                    z_asc_sel = z_asc.index_select(dim=-1, index=thres_bottom).diag()
                                    z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = (z[i, pair.begin:pair.end, :] >= z_des_exp) | (z[i, pair.begin:pair.end, :] <= z_asc_exp)
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                            case "o":
                                if multiplier.threshold == 0.0 and value == 0.0:
                                    if debug: print("Emphasis will be applied to all elements in specified tokens.")
                                    z[i, pair.begin:pair.end, :] += preoffset
                                    z[i, pair.begin:pair.end, :] *= weight
                                    z[i, pair.begin:pair.end, :] += postoffset
                                elif multiplier.threshold != 0.0 and value == 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    #z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, pair.begin:pair.end, :] < z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                elif multiplier.threshold == 0.0 and value != 0.0:
                                    thres_top = value
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    #z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=False).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = z[i, pair.begin:pair.end, :] > z_des_exp
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                                elif multiplier.threshold != 0.0 and value != 0.0:
                                    thres_top = multiplier.threshold
                                    #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                    thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                    thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                    thres_top = thres_top - 1
                                    thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    thres_bottom = value
                                    #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                    thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                    thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                    thres_bottom = thres_bottom - 1
                                    thres_bottom = torch.asarray([thres_bottom]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                    #z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                    z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                    z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    #z_asc = z_des.flip([-1])
                                    z_asc_sel = z_asc.index_select(dim=-1, index=thres_bottom).diag()
                                    z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                    target = (z[i, pair.begin:pair.end, :] < z_des_exp) & (z[i, pair.begin:pair.end, :] > z_asc_exp)
                                    if debug: print(target.to(device="cpu").nonzero().tolist())
                                    z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                    z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                    z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                            case "m":
                                thres_top = multiplier.threshold
                                thres_top = (z.shape[2] // 2) - thres_top if thres_top >= 1 else thres_top
                                thres_top = (z.shape[2] // 2) - thres_top * (z.shape[2] // 2) if thres_top < 1 else thres_top
                                thres_top = z.shape[2] if thres_top > z.shape[2] else thres_top
                                thres_top = thres_top - 1
                                thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                thres_bottom = value
                                thres_bottom = (z.shape[2] // 2) - thres_bottom if thres_bottom >= 1 else thres_bottom
                                thres_bottom = (z.shape[2] // 2) - thres_bottom * (z.shape[2] // 2) if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                thres_bottom = thres_bottom - 1
                                thres_bottom = torch.asarray([thres_bottom]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                #z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                #z_asc = z_des.flip([-1])
                                z_asc_sel = z_asc.index_select(dim=-1, index=thres_bottom).diag()
                                z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = (z[i, pair.begin:pair.end, :] < z_des_exp) & (z[i, pair.begin:pair.end, :] > z_asc_exp)
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                            case "r":
                                thres_top = multiplier.threshold
                                #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] - 1 if thres_top > z.shape[2] - 1 else thres_top
                                #thres_top = thres_top - 1
                                thres_top = torch.asarray([thres_top]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                thres_bottom = value
                                #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] - 1 if thres_bottom > z.shape[2] - 1 else thres_bottom
                                #thres_bottom = thres_bottom - 1
                                thres_bottom = torch.asarray([thres_bottom]).to(dtype=torch.int32, device=z.device).expand((pair.end-pair.begin, ))
                                #z_des = z[i, pair.begin:pair.end, :].sort(dim=-1, descending=True).values
                                z_des_sel = z_des.index_select(dim=-1, index=thres_top).diag()
                                z_des_exp = z_des_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                #z_asc = z_des
                                z_asc_sel = z_des.index_select(dim=-1, index=thres_bottom).diag()
                                z_asc_exp = z_asc_sel.unsqueeze(1).expand((-1, z.shape[2]))
                                target = (z[i, pair.begin:pair.end, :] <= z_des_exp) & (z[i, pair.begin:pair.end, :] >= z_asc_exp)
                                if debug: print(target.to(device="cpu").nonzero().tolist())
                                z[i, pair.begin:pair.end, :] += torch.where(target, preoffset, zero)
                                z[i, pair.begin:pair.end, :] *= torch.where(target, weight, one)
                                z[i, pair.begin:pair.end, :] += torch.where(target, postoffset, zero)
                            case "c":
                                thres_top = multiplier.threshold
                                #thres_top = z.shape[2] if thres_top == 0 else thres_top
                                thres_top = thres_top * z.shape[2] if thres_top < 1 else thres_top
                                thres_top = z.shape[2] - 1 if thres_top > z.shape[2] else thres_top
                                thres_top = int(thres_top)
                                #thres_top = thres_top - 1
                                thres_bottom = value
                                #thres_bottom = z.shape[2] if thres_bottom == 0 else thres_bottom
                                thres_bottom = thres_bottom * z.shape[2] if thres_bottom < 1 else thres_bottom
                                thres_bottom = z.shape[2] if thres_bottom > z.shape[2] else thres_bottom
                                # thres_bottom = thres_bottom - 1
                                thres_bottom = int(thres_bottom)
                                z[i, pair.begin:pair.end, thres_top:thres_bottom] += preoffset
                                z[i, pair.begin:pair.end, thres_top:thres_bottom] *= weight
                                z[i, pair.begin:pair.end, thres_top:thres_bottom] += postoffset
    return z

