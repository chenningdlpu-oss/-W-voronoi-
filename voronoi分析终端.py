import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- 绘图风格设置 ---
plt.style.use('ggplot')
# 自动适配中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class VoronoiMasterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voronoi Master Tool: W值计算 & 结构类型统计")
        self.root.geometry("1100x750")
        
        # 核心变量
        self.file_path = tk.StringVar()
        self.n5_col_index = tk.IntVar(value=4) # 默认n5在第5列(索引4)
        
        self.setup_ui()

    def setup_ui(self):
        # ================= 上部：设置区 =================
        top_frame = tk.Frame(self.root, pady=10)
        top_frame.pack(fill="x", padx=15)

        # 1. 标题
        tk.Label(top_frame, text="Voronoi 综合分析终端", font=("微软雅黑", 16, "bold"), fg="#333").pack(side="top", pady=(0, 10))

        # 2. 输入控件容器
        input_frame = tk.LabelFrame(top_frame, text="数据与参数配置", padx=10, pady=10)
        input_frame.pack(fill="x")

        # 第一行：文件选择
        tk.Label(input_frame, text="Voronoi 数据路径:", font=("微软雅黑", 10)).grid(row=0, column=0, sticky="w")
        tk.Entry(input_frame, textvariable=self.file_path, width=60).grid(row=0, column=1, padx=10)
        tk.Button(input_frame, text="浏览文件", command=self.select_file).grid(row=0, column=2)

        # 第二行：参数设置
        tk.Label(input_frame, text="n5 (五边形) 列索引:", font=("微软雅黑", 10)).grid(row=1, column=0, sticky="w", pady=5)
        
        idx_frame = tk.Frame(input_frame)
        idx_frame.grid(row=1, column=1, sticky="w", padx=10)
        tk.Entry(idx_frame, textvariable=self.n5_col_index, width=10).pack(side="left")
        tk.Label(idx_frame, text="(默认为 4，对应第5列)", fg="#666").pack(side="left", padx=5)

        # 3. 运行按钮
        btn = tk.Button(input_frame, text="执行分析", command=self.run_analysis,
                        bg="#007bff", fg="white", font=("Arial", 11, "bold"), width=20)
        btn.grid(row=0, column=3, rowspan=2, padx=20)

        # ================= 下部：结果区 =================
        content_frame = tk.Frame(self.root)
        content_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # 左侧：日志终端
        log_frame = tk.LabelFrame(content_frame, text="运行日志 & 数据统计", width=450)
        log_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        self.log_area = scrolledtext.ScrolledText(log_frame, font=("Courier New", 10), state='normal')
        self.log_area.pack(fill="both", expand=True)

        # 右侧：可视化图表
        self.plot_frame = tk.LabelFrame(content_frame, text="Voronoi 类型占比图 (Top 15)", width=600)
        self.plot_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))

    def log(self, msg):
        self.log_area.insert(tk.END, msg + "\n")
        self.log_area.see(tk.END)

    def select_file(self):
        f = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("Data Files", "*.dat"), ("All Files", "*.*")])
        if f: self.file_path.set(f)

    def run_analysis(self):
        # 1. 基础检查
        f_path = self.file_path.get()
        if not os.path.exists(f_path):
            messagebox.showerror("错误", "文件路径无效")
            return

        self.log_area.delete('1.0', tk.END)
        self.log(">>> 分析任务已启动...")
        
        try:
            # ================= 步骤 1: 数据加载 =================
            self.log(f"正在读取: {os.path.basename(f_path)}")
            data = np.loadtxt(f_path)
            if data.ndim == 1: data = data.reshape(1, -1)
            rows, cols = data.shape
            
            if cols < 6:
                raise ValueError(f"数据列数不足 ({cols})，无法进行 VP 统计(至少需要 n1...n6)")

            self.log(f"数据加载成功: {rows} 行 x {cols} 列")

            # ================= 步骤 2: 计算 W 值 (L5FS) =================
            # W 值是对所有原子计算的，无论其是否满足 VP 筛选
            n5_idx = self.n5_col_index.get()
            
            # 容错处理
            if n5_idx >= cols:
                raise ValueError(f"n5 索引 ({n5_idx}) 超出总列数范围")

            n5_vec = data[:, n5_idx]
            cn_vec = np.sum(data, axis=1) # 总配位数
            
            with np.errstate(divide='ignore', invalid='ignore'):
                w_values = np.true_divide(n5_vec, cn_vec)
                w_values[cn_vec == 0] = 0
            
            avg_w = np.mean(w_values)
            self.log("-" * 45)
            self.log(f"【W 值 (L5FS) 分析结果】")
            self.log(f" 体系平均 W 值 : {avg_w:.5f}")

            # ================= 步骤 3: VP 纯度筛选与统计 =================
            self.log("-" * 45)
            self.log(f"【Voronoi 类型筛选 (Filter)】")
            self.log("规则: 保留 n3,n4,n5,n6。剔除含有 n1,n2,n7... 的原子。")
            
            # (A) 低阶检查: n1(col 0), n2(col 1) 必须为 0
            mask_low = (data[:, 0] == 0) & (data[:, 1] == 0)
            
            # (B) 高阶检查: n7(col 6) 及以后必须为 0
            if cols > 6:
                sum_high = np.sum(data[:, 6:], axis=1)
                mask_high = (sum_high == 0)
            else:
                mask_high = np.ones(rows, dtype=bool)

            # (C) 综合掩码
            valid_mask = mask_low & mask_high
            valid_count = np.sum(valid_mask)
            
            self.log(f" > 有效原子数 (Valid)    : {valid_count}")
            self.log(f" > 剔除原子数 (Discarded) : {rows - valid_count}")
            self.log(f" > 有效占比 (Purity)      : {(valid_count/rows)*100:.2f}%")

            # ================= 步骤 4: 生成标签与报告 =================
            labels = []
            core_cols = data[:, 2:6] # n3, n4, n5, n6
            
            for i in range(rows):
                if valid_mask[i]:
                    # 格式化: <n3,n4,n5,n6>
                    idx = f"<{int(core_cols[i,0])},{int(core_cols[i,1])},{int(core_cols[i,2])},{int(core_cols[i,3])}>"
                    labels.append(idx)
                else:
                    labels.append("Discarded") # 被剔除

            # 统计有效标签
            valid_labels = [l for l in labels if l != "Discarded"]
            
            if len(valid_labels) == 0:
                self.log("\n❌ 警告: 没有原子满足筛选条件，无法绘制统计图。")
                return

            vp_series = pd.Series(valid_labels)
            vp_counts = vp_series.value_counts()
            vp_frac = vp_counts / valid_count * 100

            # 在日志显示 Top 10
            self.log("\n【Top 10 Voronoi Index】")
            self.log(f"{'Type':<18} {'Count':<8} {'Freq(%)':<8}")
            self.log("-" * 40)
            for typ, count in vp_counts.head(10).items():
                self.log(f"{typ:<18} {count:<8} {vp_frac[typ]:.2f}")

            # ================= 步骤 5: 保存文件 =================
            dir_name = os.path.dirname(f_path)
            base_name = os.path.splitext(os.path.basename(f_path))[0]
            
            # 整合所有数据
            df_full = pd.DataFrame(data[:, :min(cols, 10)], columns=[f'n{i+1}' for i in range(min(cols, 10))])
            if cols > 10: df_full['...'] = '...' # 示意省略
            
            df_full.insert(0, 'Atom_ID', range(rows))
            df_full['W_Value'] = w_values      # 插入 W 值
            df_full['VP_Label'] = labels       # 插入 <...> 标签
            
            # 保存全量报表
            path_full = os.path.join(dir_name, f"{base_name}_Full_Report.csv")
            df_full.to_csv(path_full, index=False)
            
            # 保存统计报表
            path_stats = os.path.join(dir_name, f"{base_name}_VP_Stats.csv")
            df_stats = pd.DataFrame({'VP_Index': vp_counts.index, 'Count': vp_counts.values, 'Fraction': vp_frac.values})
            df_stats.to_csv(path_stats, index=False)

            self.log(f"\n✅ 结果已保存至源目录:")
            self.log(f"1. {os.path.basename(path_full)} (含W值与标签)")
            self.log(f"2. {os.path.basename(path_stats)} (统计排行)")

            # ================= 步骤 6: 绘图 =================
            self.draw_chart(vp_counts.head(15), vp_frac.head(15))
            
            messagebox.showinfo("完成", "全流程分析已完成！")

        except Exception as e:
            self.log(f"\n❌ 发生错误: {str(e)}")
            messagebox.showerror("Error", str(e))

    def draw_chart(self, counts, fractions):
        # 清理旧图
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
        
        # 颜色映射：占比最高的用红色，其余用蓝色
        colors = ['#d62728' if i == 0 else '#4c72b0' for i in range(len(counts))]
        
        bars = ax.bar(range(len(counts)), fractions.values, color=colors, alpha=0.85)
        
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha='right', fontsize=9)
        
        ax.set_ylabel("Fraction (%)")
        ax.set_xlabel("Voronoi Index <n3,n4,n5,n6>")
        ax.set_title(f"Top {len(counts)} Voronoi Polyhedra Distribution")
        
        # 显示数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()

        # 嵌入 Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = VoronoiMasterApp(root)
    root.mainloop()