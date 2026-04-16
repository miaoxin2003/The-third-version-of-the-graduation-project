import numpy as np
import time
import serial  # 示例集成
import os

class Membership:
    @staticmethod
    def triangle(x, a, b, c):
        return np.maximum(np.minimum((x - a) / (b - a + 1e-6), (c - x) / (c - b + 1e-6)), 0)

class FuzzyPID:
    def __init__(self, table_path='svm/fuzzy_table.npy'):
        self.Kp_base = 0.04  # x轴示例
        self.Kd_base = 0.30
        self.Kp = self.Kp_base
        self.Kd = self.Kd_base
        self.table_path = table_path
        self.load_or_create_table()
    
    def create_table(self):
        # 3级: N/Z/P
        labels = ['N', 'Z', 'P']
        e_centers = [-30, 0, 30]  # e -50~50
        de_centers = [-12, 0, 12]  # de -20~20
        e_grid = np.linspace(-50, 50, 101)
        de_grid = np.linspace(-20, 20, 41)
        
        # 规则表 (9规则, 经验: 大|e|增Kp, 大|de|增Kd)
        rules = {
            ('N','N'): ('P','P'), ('N','Z'): ('P','Z'), ('N','P'): ('Z','P'),
            ('Z','N'): ('Z','P'), ('Z','Z'): ('Z','Z'), ('Z','P'): ('Z','P'),
            ('P','N'): ('P','P'), ('P','Z'): ('P','Z'), ('P','P'): ('Z','P'),
        }
        kp_out = {'N': -0.01, 'Z': 0, 'P': 0.01}
        kd_out = {'N': -0.05, 'Z': 0, 'P': 0.05}
        
        E, DE = np.meshgrid(e_grid, de_grid)
        table_kp = np.zeros_like(E)
        table_kd = np.zeros_like(E)
        
        memb = Membership()
        for i_de, de in enumerate(de_grid):
            for i_e, e in enumerate(e_grid):
                mu_e = {l: memb.triangle(e, c-20, c, c+20) for l,c in zip(labels, e_centers)}
                mu_de = {l: memb.triangle(de, c-8, c, c+8) for l,c in zip(labels, de_centers)}
                
                agg_kp, agg_kd, sum_mu = 0, 0, 0
                for (el, del_), (kp_l, kd_l) in rules.items():
                    mu = min(mu_e[el], mu_de[del_])
                    agg_kp += mu * kp_out[kp_l]
                    agg_kd += mu * kd_out[kd_l]
                    sum_mu += mu
                table_kp[i_de, i_e] = agg_kp / (sum_mu + 1e-6)
                table_kd[i_de, i_e] = agg_kd / (sum_mu + 1e-6)
        
        self.table = np.stack([table_kp, table_kd])  # [2,41,101]
        np.save(self.table_path, self.table)
        print('Table created & saved.')
    
    def load_or_create_table(self):
        if os.path.exists(self.table_path):
            self.table = np.load(self.table_path)
        else:
            self.create_table()
    
    def infer_fast(self, e, de):
        i_e = np.argmin(np.abs(e_grid - e))  # e_grid/de_grid global or self
        i_de = np.argmin(np.abs(de_grid - de))
        dkp = self.table[0, i_de, i_e]
        dkd = self.table[1, i_de, i_e]
        return dkp, dkd
    
    def update(self, e, de, alpha=0.1):  # alpha防震荡
        dkp, dkd = self.infer_fast(e, de)
        self.Kp = np.clip(self.Kp + alpha * dkp, 0.01, 0.1)
        self.Kd = np.clip(self.Kd + alpha * dkd, 0.1, 0.5)
        u = self.Kp * e + self.Kd * de  # PD u (Ki=0简化)
        return u, self.Kp, self.Kd

# Global grids (共享)
e_grid = np.linspace(-50,50,101)
de_grid = np.linspace(-20,20,41)

if __name__ == '__main__':
    fpid = FuzzyPID()
    print('KP/KD base:', fpid.Kp, fpid.Kd)
    
    # 实时测试
    times, es, des, us, kps = [], [], [], [], []
    t0 = time.time()
    for i in range(1000):
        e = 20 * np.sin(i*0.1) + np.random.normal(0,2)  # 仿真阶跃+噪
        de = np.diff(es + [0])[ -1] if len(es)>1 else 0
        u, kp, kd = fpid.update(e, de)
        times.append(time.time()-t0)
        es.append(e); des.append(de); us.append(u); kps.append(kp)
    
    print(f'Mean infer time: {np.mean(np.diff(times))*1000:.2f} ms')
    import matplotlib.pyplot as plt
    plt.plot(es, label='e'); plt.plot(us, label='u'); plt.legend(); plt.show()