# -*- coding: utf-8 -*-
"""
茶饮市场消费行为分析 —— Streamlit 交互式数据大屏
数据集：和鲸社区《茶饮市场消费行为数据》(31894 行 18 列)
功能：数据概览 / 数据可视化探索 / 机器学习二分类建模(高低客单预测)
"""

import os
import glob

import pandas as pd
import numpy as np
import streamlit as st

import matplotlib
matplotlib.use("Agg")  # 服务器无界面环境下使用非交互后端
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm

# ======================================================================
# 一、路径与中文字体配置（本地与云端通用，勿改动）
# ======================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "原始数据.csv")

# 自动识别字体文件（支持 .ttf 或 .otf，解决云端图表中文乱码）
_fonts = glob.glob(os.path.join(BASE_DIR, "NotoSansSC*"))
FONT_PATH = _fonts[0] if _fonts else None

if FONT_PATH and os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    _cn = fm.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams["font.sans-serif"] = [_cn, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示方块
sns.set_style("whitegrid")


# ======================================================================
# 二、数据读取与清洗（带缓存，避免每次交互重复计算）
# ======================================================================
@st.cache_data(show_spinner=False)
def load_data():
    """读取原始 CSV 并完成去重清洗，返回 去重前/去重后 两份数据。"""
    raw = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    cleaned = raw.drop_duplicates().reset_index(drop=True)
    return raw, cleaned


raw_df, df = load_data()

# 可视化用原始字符串版本（不编码，避免图表样本丢失）
df_raw = df.copy()
# 建模用编码版本（在可视化逻辑之外单独准备）
df_model = df_raw.copy()

# 常量
BRANDS = sorted(df_raw["brand"].unique().tolist())
CITY_LEVELS = ["一线", "二线", "三线"]
SCENARIOS = ["堂食", "自取", "外卖"]


# ======================================================================
# 三、绘图辅助：统一处理中文与尺寸
# ======================================================================
def _draw(fig, key, height=None):
    """把 matplotlib 图交给 Streamlit 渲染。"""
    st.pyplot(fig, clear_figure=True)


# ======================================================================
# 四、页面标题 + 侧边栏
# ======================================================================
st.set_page_config(page_title="茶饮市场消费行为分析", layout="wide",
                   page_icon="🧋")

st.sidebar.title("🧋 茶饮消费分析")
st.sidebar.caption("智能数据分析综合实训")
st.sidebar.divider()

# 数据筛选控件
selected_brands = st.sidebar.multiselect("选择品牌", BRANDS,
                                         default=BRANDS[:6])
selected_levels = st.sidebar.multiselect("选择城市等级", CITY_LEVELS,
                                         default=CITY_LEVELS)

if selected_brands:
    plot_df = df_raw[df_raw["brand"].isin(selected_brands)]
else:
    plot_df = df_raw
if selected_levels:
    plot_df = plot_df[plot_df["city_level"].isin(selected_levels)]

st.sidebar.divider()
st.sidebar.info("数据：和鲸社区·茶饮市场消费行为数据\n"
                "字段：订单/用户/地域/产品/场景五大维度，共 18 个字段")

st.title("茶饮市场消费行为分析报告")
st.caption("基于 31894 条茶饮消费订单，从城市等级、品牌、场景、人群与建模五个角度进行探索")

# ======================================================================
# 五、Tab 1：数据概览
# ======================================================================
tab_overview, tab_vis, tab_model = st.tabs(
    ["📊 数据概览", "📈 数据可视化", "🤖 机器学习模型"])

with tab_overview:
    st.subheader("数据概览与清洗说明")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("订单总数", f"{len(raw_df):,}")
    col2.metric("去重后订单数", f"{len(df):,}")
    col3.metric("重复记录数", f"{int(len(raw_df) - len(df))}")
    col4.metric("平均客单价(元)", f"{df['order_amount'].mean():.2f}")

    st.markdown(
        "**数据清洗结论**：原始数据集 31894 行、18 列；共发现 "
        f"**{int(len(raw_df) - len(df))} 条完全重复记录**，去重后为 "
        f"**{len(df):,} 行**。各字段均无缺失值。部分高值订单属于多人拼单的"
        "真实业务，不作为脏数据剔除。")
    st.markdown("**字段说明**：订单维度(order_id / quantity / order_amount / "
                "price)、用户维度(user_id / gender / age_group / member)、"
                "地域维度(city / city_level)、产品维度(brand / product_type)、"
                "场景维度(scenario)等 18 个字段。")

    st.markdown("**原始数据预览（前 500 行）**")
    st.dataframe(df_raw.head(500), width="stretch")


# ======================================================================
# 六、Tab 2：数据可视化
# ======================================================================
with tab_vis:
    st.subheader("数据可视化探索（按侧边栏筛选实时更新）")

    # ---- 1. 城市等级 ----
    st.markdown("#### 1. 城市等级与消费关系")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        cnt = plot_df["city_level"].value_counts().reindex(CITY_LEVELS)
        p = sns.barplot(x=cnt.index, y=cnt.values, hue=cnt.index,
                        palette="Set2", legend=False, ax=ax)
        ax.set_title("各城市等级订单数量分布")
        ax.set_xlabel("城市等级"); ax.set_ylabel("订单数")
        for b in p.patches:
            ax.annotate(f"{int(b.get_height())}", (b.get_x() + b.get_width()/2,
                         b.get_height()), ha="center", va="bottom")
        _draw(fig, "city_count")
    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        am = plot_df.groupby("city_level")["order_amount"].mean().reindex(CITY_LEVELS)
        p = sns.barplot(x=am.index, y=am.values, hue=am.index,
                        palette="Set2", legend=False, ax=ax)
        ax.set_title("各城市等级平均订单金额")
        ax.set_xlabel("城市等级"); ax.set_ylabel("平均订单金额(元)")
        for b in p.patches:
            ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width()/2,
                         b.get_height()), ha="center", va="bottom")
        _draw(fig, "city_amount")
    st.caption("结论：二线城市订单最多，但各城市等级客单价接近(一线42.83/二线42.68/"
                "三线43.20元)，城市等级对客单价影响有限。")

    # ---- 2. 品牌 ----
    st.markdown("#### 2. 品牌销量与客单价对比")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        bc = plot_df["brand"].value_counts()
        p = sns.barplot(x=bc.values, y=bc.index, hue=bc.index,
                        palette="Set2", legend=False, ax=ax)
        ax.set_title("各品牌订单数量排名")
        ax.set_xlabel("订单数")
        _draw(fig, "brand_count")
    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ba = plot_df.groupby("brand")["order_amount"].mean().sort_values(ascending=False)
        p = sns.barplot(x=ba.values, y=ba.index, hue=ba.index,
                        palette="Set2", legend=False, ax=ax)
        ax.set_title("各品牌平均订单金额排名")
        ax.set_xlabel("平均订单金额(元)")
        _draw(fig, "brand_amount")
    st.caption("结论：平价品牌(蜜雪冰城)走销量，中高端品牌(CoCo、古茗等)客单价更高，"
                "订单量与客单价呈反向关系。")

    # ---- 3. 消费场景 ----
    st.markdown("#### 3. 消费场景分布与金额差异")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sc = plot_df["scenario"].value_counts().reindex(SCENARIOS)
        ax.pie(sc.values, labels=SCENARIOS, autopct="%.1f%%",
               colors=["#b8c8d8", "#c8d8bc", "#e2d2d0"])
        ax.set_title("消费场景分布占比")
        _draw(fig, "scenario_pie")
    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sa = plot_df.groupby("scenario")["order_amount"].mean().reindex(SCENARIOS)
        p = sns.barplot(x=sa.index, y=sa.values, hue=sa.index,
                        palette="Set2", legend=False, ax=ax)
        ax.set_title("不同消费场景平均订单金额")
        ax.set_xlabel("消费场景"); ax.set_ylabel("平均订单金额(元)")
        for b in p.patches:
            ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width()/2,
                         b.get_height()), ha="center", va="bottom")
        _draw(fig, "scenario_amount")
    st.caption("结论：堂食占比 64.9% 最高，消费以到店为主；各场景客单价差异不大。")

    # ---- 4. 年龄段 ----
    st.markdown("#### 4. 年龄段与消费能力")
    fig, ax = plt.subplots(figsize=(8, 4))
    age_order = ["18-24岁", "25-34岁", "35-44岁", "45岁以上"]
    aa = plot_df.groupby("age_group")["order_amount"].mean().reindex(age_order)
    p = sns.barplot(x=aa.index, y=aa.values, hue=aa.index,
                    palette="Set2", legend=False, ax=ax)
    ax.set_title("不同年龄段平均订单金额")
    ax.set_xlabel("年龄段"); ax.set_ylabel("平均订单金额(元)")
    for b in p.patches:
        ax.annotate(f"{b.get_height():.2f}元", (b.get_x() + b.get_width()/2,
                     b.get_height()), ha="center", va="bottom")
    _draw(fig, "age_amount")
    st.caption("结论：各年龄段客单价非常接近(约 42.7~43.2 元)，年龄段对客单价影响微弱。")

    # ---- 5. 会员对比 ----
    st.markdown("#### 5. 会员与非会员消费差异")
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ma = plot_df.groupby("member")["order_amount"].mean()
    mq = plot_df.groupby("member")["quantity"].mean()
    x_pos = np.arange(2); width = 0.35
    b1 = ax1.bar(x_pos - width/2, ma.values, width, label="平均订单金额", color="#8899bb")
    ax1.set_ylabel("平均订单金额(元)"); ax1.set_xticks(x_pos); ax1.set_xticklabels(["非会员", "会员"])
    for b in b1:
        ax1.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width()/2,
                     b.get_height()), ha="center", va="bottom")
    ax2 = ax1.twinx()
    b2 = ax2.bar(x_pos + width/2, mq.values, width, label="平均购买数量", color="#bb9988")
    ax2.set_ylabel("平均购买数量")
    for b in b2:
        ax2.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width()/2,
                     b.get_height()), ha="center", va="bottom")
    l1, lab1 = ax1.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper left")
    ax1.set_title("会员与非会员消费对比")
    _draw(fig, "member")
    st.caption("结论：会员客单价(42.87元)与购买数量(2.01)略高于非会员，会员对消费有微弱正向促进。")

    # ---- 6. 订单金额分布 ----
    st.markdown("#### 6. 订单金额分布直方图")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(plot_df["order_amount"], bins=30, color="#8899bb", edgecolor="white", ax=ax)
    ax.set_title("订单金额分布直方图")
    ax.set_xlabel("订单金额(元)"); ax.set_ylabel("订单数量")
    _draw(fig, "hist")
    st.caption("结论：订单金额右偏分布，集中在 20-40 元区间，高价订单占比少。")

    # ---- 7. 箱线图 ----
    st.markdown("#### 7. 订单金额与购买数量箱线图(异常值检测)")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(y=plot_df["order_amount"], palette="Set2", ax=ax)
        ax.set_title("订单金额箱线图"); ax.set_ylabel("订单金额(元)")
        _draw(fig, "box_amount")
    with c2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(y=plot_df["quantity"], palette="Set2", ax=ax)
        ax.set_title("购买数量箱线图"); ax.set_ylabel("购买数量")
        _draw(fig, "box_qty")
    st.caption("结论：箱体集中在 20-60 元区间，与茶饮行业常规消费价格区间一致。")


# ======================================================================
# 七、Tab 3：机器学习模型（二分类预测高低客单）
# ======================================================================
with tab_model:
    st.subheader("机器学习建模：预测订单是否为高客单")

    st.markdown(
        "**任务说明**：直接回归预测订单精确金额效果差(负R²)，故调整为**二分类任务**。"
        "以整体平均订单金额 42 元为阈值，`order_amount > 42` 标记为高客单(1)。"
        "输入特征为用户/地域/场景特征，**不使用** price、quantity 以避免特征泄露。")

    # 相关性热力图（基于全量去重数据）
    st.markdown("#### 特征相关性热力图")
    features = ["city_level", "product_type", "scenario", "gender",
                "age_group", "member", "order_amount"]

    @st.cache_data(show_spinner=False)
    def prep_model_data():
        m = df_model.copy()
        m["city_level"] = m["city_level"].str.strip().map({"一线": 0, "二线": 1, "三线": 2})
        m["product_type"] = m["product_type"].str.strip().map({
            "低糖茶饮": 0, "轻乳茶": 1, "燕麦茶": 2, "水果茶": 3, "奶茶/芝士/其他": 4})
        m["scenario"] = m["scenario"].str.strip().map({"堂食": 0, "自取": 1, "外卖": 2})
        m["gender"] = m["gender"].str.strip().map({"男": 0, "女": 1})
        m["age_group"] = m["age_group"].str.strip().map({
            "18-24岁": 0, "25-34岁": 1, "35-44岁": 2, "45岁以上": 3})
        m["high_customer"] = (m["order_amount"] > 42).astype(int)
        return m

    mdata = prep_model_data()

    fig, ax = plt.subplots(figsize=(8, 6))
    corr = mdata[features].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("特征相关性热力图")
    _draw(fig, "corr")
    st.caption("结论：用户/地域/场景特征与订单金额的线性相关系数均接近 0，"
                "变量间几乎不存在简单线性关系。")

    # 训练模型
    st.markdown("#### 模型训练与对比")

    @st.cache_data(show_spinner=True)
    def train_models():
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report

        fmodel = ["city_level", "product_type", "scenario", "gender", "age_group", "member"]
        X = mdata[fmodel]
        y = mdata["high_customer"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # 逻辑回归
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(X_train); Xte_s = scaler.transform(X_test)
        lr = LogisticRegression(random_state=42, max_iter=500)
        lr.fit(Xtr_s, y_train)
        y_lr = lr.predict(Xte_s)
        acc_lr = accuracy_score(y_test, y_lr)
        rep_lr = classification_report(y_test, y_lr, output_dict=True, zero_division=0)

        # 随机森林
        rf = RandomForestClassifier(n_estimators=80, max_depth=8,
                                    min_samples_leaf=4, random_state=42)
        rf.fit(X_train, y_train)
        y_rf = rf.predict(X_test)
        acc_rf = accuracy_score(y_test, y_rf)
        rep_rf = classification_report(y_test, y_rf, output_dict=True, zero_division=0)
        acc_rf_train = accuracy_score(y_train, rf.predict(X_train))

        importance = pd.DataFrame({"特征": fmodel,
                                   "重要性": rf.feature_importances_}).sort_values(
            "重要性", ascending=False)
        return (acc_lr, rep_lr, acc_rf, rep_rf, acc_rf_train, importance)

    acc_lr, rep_lr, acc_rf, rep_rf, acc_rf_train, importance = train_models()

    m1, m2, m3 = st.columns(3)
    m1.metric("逻辑回归 Accuracy", f"{acc_lr*100:.2f}%")
    m2.metric("随机森林 Accuracy", f"{acc_rf*100:.2f}%")
    m3.metric("随机森林 训练集 Accuracy", f"{acc_rf_train*100:.2f}%")

    st.markdown("**逻辑回归 分类报告**")
    st.dataframe(pd.DataFrame(rep_lr).T.round(4), width="stretch")
    st.markdown("**随机森林 分类报告**")
    st.dataframe(pd.DataFrame(rep_rf).T.round(4), width="stretch")

    st.markdown("#### 随机森林特征重要性排名")
    fig, ax = plt.subplots(figsize=(7, 4))
    p = sns.barplot(data=importance, x="重要性", y="特征", hue="特征",
                    palette="Set2", legend=False, ax=ax)
    ax.set_title("随机森林分类——特征重要性排名")
    _draw(fig, "importance")

    st.caption(
        "结论：二分类后逻辑回归准确率约 57%、随机森林约 56%，模型略优于随机猜测但整体能力"
        "有限；训练/测试集准确率接近，无明显过拟合。产品类型(product_type)是最重要特征。"
        "说明仅靠用户画像+地域场景特征区分高低客单订单能力有限，还需补充品牌、价格档位等信息。")
#（注：内容由AI生成）
