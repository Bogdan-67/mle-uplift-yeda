import matplotlib.pyplot as plt
import numpy as np
from sklift.metrics import uplift_by_percentile
import mlflow

def custom_uplift_by_percentile(y_true, uplift, treatment, 
                               kind='line', bins=10, string_percentiles=True, 
                               figsize=(10, 6), title=None):
    """
    Построение графика uplift по перцентилям.
    
    Аргументы:
        y_true: Бинарные целевые значения
        uplift: Прогнозируемые значения uplift
        treatment: Бинарные индикаторы воздействия
        kind: 'line' или 'bar'
        bins: Количество перцентильных корзин
        string_percentiles: Отображать ли перцентили в виде строк
        figsize: Размер рисунка (кортеж)
        title: Пользовательский заголовок для графика
    
    Возвращает:
        Рисунок matplotlib
    """
    
    # получаем данные по перцентилям, используя функцию из sklift
    df = uplift_by_percentile(
        y_true, uplift, treatment, strategy='overall',
        std=True, total=False, bins=bins, string_percentiles=False
    )
    
    # извлекаем перцентили из индекса DataFrame
    percentiles = df.index[:bins].values.astype(float)
    
    # извлекаем значения отклика для тестовой группы и их стандартные отклонения
    response_rate_trmnt = df.loc[percentiles, 'response_rate_treatment'].values
    std_trmnt = df.loc[percentiles, 'std_treatment'].values
    
    # извлекаем значения отклика для контрольной группы и их стандартные отклонения
    response_rate_ctrl = df.loc[percentiles, 'response_rate_control'].values
    std_ctrl = df.loc[percentiles, 'std_control'].values
    
    # извлекаем значения uplift и их стандартные отклонения
    uplift_score = df.loc[percentiles, 'uplift'].values
    std_uplift = df.loc[percentiles, 'std_uplift'].values
    
    # создаём график
    fig, ax = plt.subplots(figsize=figsize)
    
    if kind == 'line':
        # строим линейный график для тестовой группы с погрешностями
        ax.errorbar(
            percentiles, response_rate_trmnt, yerr=std_trmnt,
            linewidth=2, color='forestgreen', label='Отклик тестовой группы'
        )
        # строим линейный график для контрольной группы с погрешностями
        ax.errorbar(
            percentiles, response_rate_ctrl, yerr=std_ctrl,
            linewidth=2, color='orange', label='Отклик контрольной группы'
        )
        # строим линейный график для uplift с погрешностями
        ax.errorbar(
            percentiles, uplift_score, yerr=std_uplift,
            linewidth=2, color='red', label='Uplift'
        )
        # заполняем область между линиями тестовой и контрольной групп
        ax.fill_between(percentiles, response_rate_trmnt,
                        response_rate_ctrl, alpha=0.1, color='red')
        
        # добавляем горизонтальную линию на уровне 0, если есть отрицательные значения uplift
        if np.amin(uplift_score) < 0:
            ax.axhline(y=0, color='black', linewidth=1)
            
    elif kind == 'bar':
        # вычисляем ширину столбцов для столбчатой диаграммы
        width = percentiles[1] - percentiles[0] if len(percentiles) > 1 else 5
        bar_width = width * 0.35
        
        # строим столбцы для тестовой, контрольной групп и для uplift
        ax.bar(percentiles - bar_width, response_rate_trmnt, bar_width, 
               color='forestgreen', label='Отклик тестовой группы')
        ax.bar(percentiles, response_rate_ctrl, bar_width, 
               color='orange', label='Отклик контрольной группы')
        ax.bar(percentiles + bar_width, uplift_score, bar_width, 
               color='red', label='Uplift')
    
    # устанавливаем метки по оси X
    if string_percentiles:
        # создаём строковые метки для перцентилей (диапазоны)
        percentiles_str = [f"0-{percentiles[0]:.0f}"] + \
                          [f"{percentiles[i]:.0f}-{percentiles[i + 1]:.0f}" 
                           for i in range(len(percentiles) - 1)]
        ax.set_xticks(percentiles)
        ax.set_xticklabels(percentiles_str, rotation=45)
    else:
        # используем числовые значения перцентилей
        ax.set_xticks(percentiles)
    
    # устанавливаем подписи осей и заголовок
    ax.set_xlabel('Перцентиль')
    ax.set_ylabel('Уровень отклика / Uplift')
    
    # устанавливаем заголовок, если он предоставлен
    if title:
        ax.set_title(title)
  
    # добавляем легенду и сетку для улучшения читаемости
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # оптимизируем расположение элементов на графике
    plt.tight_layout()
    return fig


def get_or_create_run_id(study, experiment_id, name = 'optuna_param_search'):
    """
        Функция для получения run_id MLflow из пользовательского поля в бд optuna
        Если поле не задано, то запускается новый run и записывается в бд
    """
    run_id = study.user_attrs.get("mlflow_run_id")
    
    if run_id:
        try:
            # Проверяем, существует ли запуск и в том ли он эксперименте
            existing_run = mlflow.get_run(run_id)
            if existing_run.info.experiment_id == experiment_id:
                return run_id
            else:
                print("Найден run_id от другого эксперимента, создаем новый.")
        except:
            print("Run_id не найден в MLflow, создаем новый.")
    
    with mlflow.start_run(run_name=name, experiment_id=experiment_id) as run:
        new_id = run.info.run_id
        study.set_user_attr("mlflow_run_id", new_id)
        return new_id