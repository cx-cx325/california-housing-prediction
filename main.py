import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为 SimHei
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 缓存数据加载
@st.cache_data
def load_data():
    """加载加州房价数据集并进行异常值处理"""
    data_path = r'C:\Users\A\Desktop\predict_price\data\CaliforniaHousing\cal_housing.data'  # 确保路径正确
    column_names = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
        'Latitude', 'Longitude', 'MedHouseVal'
    ]
    df = pd.read_csv(data_path, sep=',', header=None, names=column_names)
    
    # 异常值处理：分位数截断法（1%-99%）
    for column in df.columns:
        q1 = df[column].quantile(0.01)
        q99 = df[column].quantile(0.99)
        df[column] = df[column].clip(lower=q1, upper=q99)
    
    return df

def plot_data(df):
    """绘制房价分布图"""
    st.subheader("房价分布图")
    fig, ax = plt.subplots()
    ax.hist(df['MedHouseVal'], bins=30, edgecolor='black')
    ax.set_xlabel("房价 (单位: $100,000)")
    ax.set_ylabel("频数")
    ax.set_title("加州房价分布")
    st.pyplot(fig)

def plot_boxplot(df):
    """绘制箱线图并标注异常值比例"""
    st.subheader("箱线图")
    fig, ax = plt.subplots(figsize=(12, 8))  # 调整图表大小

    # 绘制每个特征的箱线图
    boxplot = df.boxplot(ax=ax, rot=45)
    ax.set_title("各特征的箱线图")
    ax.set_xlabel("特征")
    ax.set_ylabel("值")

    # 计算并标注每个特征的异常值比例
    for i, column in enumerate(df.columns):
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
        outlier_ratio = len(outliers) / len(df) * 100
        ax.text(i+1, df[column].max(), f'{outlier_ratio:.1f}%', ha='center', fontsize=8)

    # 添加说明
    plt.figtext(0.5, 0.01, "图中圆点表示异常值，方框代表25%-75%分位数，上方标注为异常值比例。", ha='center', fontsize=10)
    st.pyplot(fig)

def build_model(df):
    """构建随机森林回归模型"""
    st.subheader("模型构建")

    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, X_test, y_test, y_pred

def build_and_compare_models(df):
    """构建并比较不同模型"""
    st.subheader("模型对比")

    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 定义模型
    models = {
        '线性回归': LinearRegression(),
        '支持向量机': SVR(),
        '随机森林': RandomForestRegressor(n_estimators=100, random_state=42),
        '随机森林变体1': RandomForestRegressor(n_estimators=200, max_depth=30, random_state=42),
        '随机森林变体2': RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_split=2, random_state=42),
        '随机森林变体3': RandomForestRegressor(n_estimators=150, max_depth=25, min_samples_split=5, random_state=42),
        '随机森林变体4': ExtraTreesRegressor(n_estimators=100, random_state=42),
        '随机森林变体5': ExtraTreesRegressor(n_estimators=150, max_depth=30, random_state=42)
    }

    # 训练模型并计算性能指标
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append((name, mse, r2))

    # 创建DataFrame存储结果
    results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'R²'])

    # 绘制对比图
    fig, ax1 = plt.subplots(figsize=(12, 6))  # 调整图表大小

    # 绘制MSE柱状图
    ax1.bar(results_df['Model'], results_df['MSE'], color='b', alpha=0.6, label='均方误差')
    ax1.set_xlabel('模型', fontsize=10)  # 调整字体大小
    ax1.set_ylabel('均方误差 (单位: $100,000²)', fontsize=10)  # 添加单位说明
    ax1.tick_params(axis='y', labelcolor='b', labelsize=9)
    ax1.tick_params(axis='x', rotation=45, labelsize=9)  # 旋转横轴标签

    # 创建第二个y轴
    ax2 = ax1.twinx()
    ax2.plot(results_df['Model'], results_df['R²'], color='r', marker='o', label='决定系数')
    ax2.set_ylabel('决定系数', color='r', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='r', labelsize=9)

    # 添加标题和图例
    plt.title('模型性能对比', fontsize=12)
    fig.tight_layout()
    st.pyplot(fig)

    return results_df

def plot_feature_importance(model, df):
    """绘制特征重要性图"""
    st.subheader("特征重要性图")

    feature_names = df.columns[:-1]
    importances = model.feature_importances_

    # 创建DataFrame存储特征重要性
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # 绘制水平柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    ax.set_xlabel('重要性')
    ax.set_title('特征重要性')
    st.pyplot(fig)

    return importance_df

def show_data_statistics(df):
    """显示数据集的统计描述表格"""
    st.subheader("数据集的统计描述")

    # 生成统计描述
    statistics = df.describe()

    # 显示表格
    st.write(statistics)

def user_input_features(df):
    """用户输入特征值"""
    st.sidebar.header("用户输入特征值")
    features = {}
    for col in df.columns[:-1]:
        col_min = df[col].min()  # 动态计算最小值
        col_max = df[col].max()  # 动态计算最大值
        features[col] = st.sidebar.slider(
            f"输入 {col}",
            float(col_min),
            float(col_max),
            float(df[col].mean())
        )
    return pd.DataFrame([features])

def plot_results(y_test, y_pred, X_test):
    """绘制实际值与预测值的对比图，并根据经纬度添加颜色映射"""
    st.subheader("实际值与预测值对比")
    fig, ax = plt.subplots()
    # 根据经纬度计算颜色值
    colors = X_test['Latitude'] + X_test['Longitude']
    scatter = ax.scatter(y_test, y_pred, c=colors, cmap='viridis', edgecolor='black')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=1)
    ax.set_xlabel("实际房价")
    ax.set_ylabel("预测房价")
    ax.set_title("实际值与预测值对比")
    # 添加颜色条
    plt.colorbar(scatter, ax=ax, label='经纬度综合值')
    st.pyplot(fig)

def plot_feature_correlation(df):
    """绘制特征相关性热力图"""
    st.subheader("特征相关性热力图")
    
    # 计算相关系数矩阵
    correlation_matrix = df.corr()
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("特征相关性矩阵")
    st.pyplot(fig)
    
    # 解释特征筛选依据
    st.markdown("""
    ### 特征筛选依据：
    1. **相关性分析**：通过相关性热力图，可以直观地看到各特征与目标变量（房价）之间的相关性。相关性较高的特征对模型的预测能力贡献更大。
    2. **多重共线性**：避免选择相关性过高的特征，以减少多重共线性对模型的影响。
    3. **特征重要性**：结合随机森林模型的特征重要性分析，筛选出对模型预测贡献最大的特征。
    """)

def plot_parameter_tuning(df):
    """绘制参数调优过程图"""
    st.subheader("参数调优过程图")
    
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 定义参数网格
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 10, 20, 30],
    }
    
    # 初始化模型
    model = RandomForestRegressor(random_state=42)
    
    # 网格搜索
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)
    
    # 获取结果
    results = pd.DataFrame(grid_search.cv_results_)
    
    # 绘制n_estimators对MSE的影响
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    for max_depth in param_grid['max_depth']:
        filtered_results = results[results['param_max_depth'] == max_depth]
        ax1.plot(filtered_results['param_n_estimators'], -filtered_results['mean_test_score'], label=f'max_depth={max_depth}')
    ax1.set_xlabel('n_estimators')
    ax1.set_ylabel('MSE (单位: $100,000²)')
    ax1.set_title('不同n_estimators对MSE的影响')
    ax1.legend()
    st.pyplot(fig1)
    
    # 绘制max_depth对MSE的影响
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for n_estimators in param_grid['n_estimators']:
        filtered_results = results[results['param_n_estimators'] == n_estimators]
        ax2.plot(filtered_results['param_max_depth'].astype(str), -filtered_results['mean_test_score'], label=f'n_estimators={n_estimators}')
    ax2.set_xlabel('max_depth')
    ax2.set_ylabel('MSE (单位: $100,000²)')
    ax2.set_title('不同max_depth对MSE的影响')
    ax2.legend()
    st.pyplot(fig2)
    
    # 解释参数选择依据
    st.markdown("""
    ### 参数选择依据：
    1. **n_estimators**：通过调整决策树的数量，找到既能提高模型性能又不会过度增加计算复杂度的最优值。
    2. **max_depth**：控制决策树的最大深度，避免过拟合，同时确保模型能够捕捉到数据中的复杂模式。
    3. **交叉验证**：使用5折交叉验证评估不同参数组合下的模型性能，选择平均MSE最小的参数组合。
    """)

def plot_time_series(df):
    """绘制房价随时间变化的折线图"""
    st.subheader("房价时间趋势分析")
    
    # 按房屋建造年份排序
    time_sorted_df = df.sort_values(by='HouseAge')
    
    # 绘制房价随时间变化的折线图
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_sorted_df['HouseAge'], time_sorted_df['MedHouseVal'], label='实际房价')
    
    # 添加模型预测曲线
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X)
    ax.plot(time_sorted_df['HouseAge'], y_pred[time_sorted_df.index], label='预测房价', linestyle='--')
    
    ax.set_xlabel("房屋建造年份")
    ax.set_ylabel("房价 (单位: $100,000)")
    ax.set_title("房价随时间变化趋势")
    ax.legend()
    st.pyplot(fig)

def main():
    st.title("加州房价预测")
    
    try:
        df = load_data()
        
        if df is None or df.empty:
            st.error("未能加载数据集。请检查数据路径和文件格式。")
            return
        
        if st.checkbox("显示数据集"):
            st.write(df)
        
        plot_data(df)
        plot_boxplot(df)
        
        # 新增特征相关性热力图
        plot_feature_correlation(df)
        
        model, X_test, y_test, y_pred = build_model(df)
        
        input_df = user_input_features(df)
        if st.sidebar.button("预测房价"):
            prediction = model.predict(input_df.values)
            st.sidebar.success(f"预测房价为: {prediction[0]:.2f} (单位: $100,000)")
        
        plot_results(y_test, y_pred, X_test)
        
        show_data_statistics(df)
        plot_feature_importance(model, df)
        build_and_compare_models(df)
        
        # 新增参数调优过程图
        plot_parameter_tuning(df)
        
        # 新增时间序列分析模块
        plot_time_series(df)
        
    except Exception as e:
        st.error(f"发生错误: {e}")

if __name__ == "__main__":
    main()