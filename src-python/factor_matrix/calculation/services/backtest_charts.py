"""Presentation-only charts for already computed prices, fills and account paths.

The horizontal axis is an explicit session axis, not an intraday timeline.
Daily holdings use bars; no smooth interpolation or invented observations.
Matplotlib is optional and imported only when rendering.
"""
from datetime import datetime
from math import ceil
from pathlib import Path


def render_backtest_charts(prices: list[dict], fills: list[dict], path_metrics: dict,
                          positions: list[dict], *, directory: Path, title: str,
                          data_label: str, language: str = 'en',
                          font_path: Path | None = None,
                          daily_accounts: list[dict] | None = None,
                          corporate_actions: list[dict] | None = None) -> list[Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib import font_manager

    zh = language == 'zh'
    def label(chinese, english):
        return chinese if zh else english

    style = {'svg.hashsalt': 'factor-backtest-charts-v2', 'font.size': 10,
             'axes.unicode_minus': False, 'axes.titleweight': 'semibold',
             'text.color': '#172b43', 'axes.labelcolor': '#607085',
             'xtick.color': '#607085', 'ytick.color': '#607085'}
    if font_path is not None:
        font_manager.fontManager.addfont(str(font_path))
        style['font.family'] = font_manager.FontProperties(fname=str(font_path)).get_name()

    curve = path_metrics['curve']
    assets = sorted({r['asset_id'] for r in prices})
    sessions = sorted({datetime.fromisoformat(r['time']).date().isoformat() for r in curve})
    if not assets or not sessions:
        raise ValueError('CHART_REQUIRES_PRICES_AND_NAV')
    session_index = {day: i for i, day in enumerate(sessions)}
    def day_index(timestamp):
        return session_index[datetime.fromisoformat(timestamp).date().isoformat()]

    # This is a daily report. Reject ambiguous intraday NAV instead of silently
    # coalescing it; the original event timestamps remain in source artifacts.
    if len(sessions) != len(curve):
        raise ValueError('CHART_REQUIRES_ONE_NAV_PER_SESSION')
    price_rows = ceil(len(assets) / 2)
    height = 7.8 + 2.5 * price_rows
    palette = ['#3566a3', '#8661a6', '#b48836', '#467d79']
    with plt.rc_context(style):
        figure = plt.figure(figsize=(13.2, height), facecolor='#f7f9fc')
        grid = figure.add_gridspec(price_rows + 2, 2, left=.075, right=.965,
                                  bottom=.10, top=.735, hspace=.67, wspace=.25,
                                  height_ratios=[1.05] + [1] * price_rows + [.8])
        nav_ax, dd_ax = figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])
        price_axes = [figure.add_subplot(grid[1+i//2, i%2]) for i in range(len(assets))]
        holding_ax = figure.add_subplot(grid[-1, :])
        axes = [nav_ax, dd_ax, *price_axes, holding_ax]
        figure.text(.075, .949, title, fontsize=23, weight='bold')
        figure.text(.075, .911, data_label, fontsize=11, color='#986224',
                    bbox={'facecolor': '#fff2de', 'edgecolor': 'none', 'pad': 7})
        figure.text(.075, .875,
                    f"{sessions[0]} — {sessions[-1]}  ·  {len(sessions)} "
                    + label('个日终观测；日期为样本标签' if 'fixture' in data_label.lower() or '合成' in data_label else '个日终观测',
                            'daily observations'), fontsize=10, color='#607085')
        metrics = [
            (label('期初资金 / 元', 'INITIAL NAV / CNY'), f"{float(path_metrics['initial_nav']):,.2f}", '#172b43'),
            (label('期末净值 / 元', 'FINAL NAV / CNY'), f"{float(curve[-1]['nav']):,.2f}", '#172b43'),
            (label('最大回撤', 'MAXIMUM DRAWDOWN'), f"{float(path_metrics['max_drawdown']):.2%}", '#b74853'),
            (label('模拟成交笔数', 'SIMULATED FILLS'), str(len(fills)), '#172b43'),
        ]
        for x, (name, value, color) in zip([.075, .31, .545, .78], metrics):
            figure.text(x, .827, name, fontsize=10, color='#607085')
            figure.text(x, .789, value, fontsize=23, weight='bold', color=color)

        x = [day_index(r['time']) for r in curve]
        nav = [float(r['nav']) for r in curve]
        drawdown = [-float(r['drawdown']) for r in curve]
        nav_ax.plot([-1, *x], [float(path_metrics['initial_nav']), *nav], color='#3566a3', linewidth=2, marker='o', markersize=4)
        nav_ax.axhline(float(path_metrics['initial_nav']), color='#acb6c4', linestyle='--', linewidth=1,
                       label=label('期初资金', 'Initial capital'))
        nav_ax.set_title(label('01  账户净值', '01  Account NAV'), loc='left', pad=12)
        nav_ax.set_ylabel(label('元', 'CNY'))
        nav_ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
        nav_ax.margins(y=.18)
        nav_ax.legend(frameon=False, fontsize=9, loc='upper right')
        nav_ax.annotate(f'{nav[-1]:,.0f}', (x[-1], nav[-1]), xytext=(-8, 10),
                        textcoords='offset points', ha='right', color='#3566a3', weight='bold')
        for account in daily_accounts or []:
            flow = float(account.get('external_flow', 0))
            if flow:
                nav_ax.annotate(label('外部入金' if flow > 0 else '外部出金', 'External cash flow')+f' {flow:+,.0f}',
                    (day_index(account['time']), float(account['nav'])), xytext=(0, 20),
                    textcoords='offset points', ha='center', fontsize=9, color='#986224')

        dd_ax.axhline(0, color='#acb6c4', linewidth=1)
        dd_ax.fill_between([-1, *x], [0, *drawdown], 0, color='#e7a7ad', alpha=.4)
        dd_ax.plot([-1, *x], [0, *drawdown], color='#b74853', linewidth=1.8, marker='o', markersize=4)
        dd_ax.set_title(label('02  回撤 · 剔除外部资金流影响', '02  Flow-neutral drawdown'), loc='left', pad=12)
        dd_ax.yaxis.set_major_formatter(mticker.PercentFormatter(1, decimals=1))
        dd_ax.set_ylim(min(drawdown) - max(.01, abs(min(drawdown))*.25), .006)
        trough = path_metrics.get('max_drawdown_trough_time')
        if trough:
            tx = day_index(trough)
            ty = -float(path_metrics['max_drawdown'])
            dd_ax.scatter([tx], [ty], color='#b74853', s=42, zorder=4)
            dd_ax.annotate(f'{ty:.2%}', (tx, ty), xytext=(0, -17),
                           textcoords='offset points', ha='center', color='#b74853', weight='bold')

        for i, (ax, asset) in enumerate(zip(price_axes, assets)):
            color = palette[i % len(palette)]
            rows = sorted((r for r in prices if r['asset_id'] == asset), key=lambda r: r['time'])
            ys = [float(r['close']) for r in rows]
            ax.plot([day_index(r['time']) for r in rows], ys, color=color, linewidth=1.8,
                    marker='o', markersize=3, label=label('收盘价', 'Close'))
            for side, marker, fill_color in [('buy', '^', '#15816a'), ('sell', 'v', '#b74853')]:
                trades = [r for r in fills if r['asset_id'] == asset and r['side'] == side]
                if not trades:
                    continue
                side_label = label('买入' if side == 'buy' else '卖出', side.capitalize())
                ax.scatter([day_index(r['time']) for r in trades], [float(r['price']) for r in trades],
                           marker=marker, color=fill_color, s=85, edgecolors='white',
                           linewidths=.8, zorder=5, label=side_label)
                for trade in trades:
                    ys.append(float(trade['price']))
                    ax.annotate(side_label + f" {trade['quantity']}",
                                (day_index(trade['time']), float(trade['price'])),
                                xytext=(0, 13 if side == 'buy' else -22), textcoords='offset points',
                                ha='center', fontsize=9, color=fill_color)
            padding = max((max(ys)-min(ys))*.35, abs(max(ys))*.015, .05)
            ax.set_ylim(min(ys)-padding, max(ys)+padding)
            for action in corporate_actions or []:
                if action['asset_id'] != asset or action['kind'] not in ('split', 'dividend_record'):
                    continue
                name = (label('拆分', 'Split')+f" {action['denominator']}→{action['numerator']}" if action['kind'] == 'split'
                        else label('除息', 'Ex-dividend')+f" {float(action['cash_per_share']):.2f}")
                ax.axvline(day_index(action['time']), color='#acb6c4', linestyle=':', linewidth=1)
                ax.text(day_index(action['time'])+.08, .08 if action['kind'] == 'split' else .24,
                        name, transform=ax.get_xaxis_transform(), fontsize=8, color='#607085')
            ax.set_title(f'{i+3:02d}  {asset}  ·  '+label('价格与成交', 'Price and fills'), loc='left', pad=12)
            ax.set_ylabel(label('元 / 股', 'CNY / share'))
            ax.legend(frameon=False, fontsize=8, loc='upper right', ncol=3)

        if daily_accounts is not None:
            accounts = {r['date']: r for r in daily_accounts}
            if any(day not in accounts for day in sessions):
                raise ValueError('STACK_REQUIRES_ALL_DAILY_ACCOUNTS')
            values = {(r['date'], r['asset_id']): float(r['market_value']) for r in positions}
            bottoms = [0.] * len(sessions)
            for i, asset in enumerate(assets):
                ys = [values.get((day, asset), 0) for day in sessions]
                holding_ax.bar(x, ys, bottom=bottoms, width=.7, color=palette[i % len(palette)], label=asset, zorder=3)
                bottoms = [a+b for a,b in zip(bottoms, ys)]
            for key, color, name in [('cash', '#b8cbbd', label('现金', 'Cash')), ('receivables', '#d5b878', label('应收股息', 'Receivables'))]:
                ys = [float(accounts[day][key]) for day in sessions]
                holding_ax.bar(x, ys, bottom=bottoms, width=.7, color=color, label=name, zorder=3)
                bottoms = [a+b for a,b in zip(bottoms, ys)]
            if any(abs(v-float(accounts[d]['nav'])) > .005 for v,d in zip(bottoms, sessions)):
                raise ValueError('STACK_DOES_NOT_RECONCILE_NAV')
            holding_ax.bar([-1], [float(path_metrics['initial_nav'])], width=.7, color='#b8cbbd', zorder=3)
            holding_ax.set_title(label('资产构成 · 市值＋现金＋应收款＝净值', 'Assets · market value + cash + receivables = NAV'), loc='left', pad=12)
            holding_ax.set_ylabel(label('元', 'CNY'))
        else:
            values = {(r['date'], r['asset_id']): r['quantity'] for r in positions}
            width = .7 / len(assets)
            for i, asset in enumerate(assets):
                holding_ax.bar([j+(i-(len(assets)-1)/2)*width for j in range(len(sessions))],
                               [values.get((day, asset), 0) for day in sessions], width=width*.9,
                               color=palette[i % len(palette)], label=asset, zorder=3)
            holding_ax.set_title(label('日终持仓股数 · 旧版报表', 'Share counts · legacy report'), loc='left', pad=12)
            holding_ax.set_ylabel(label('股', 'Shares'))
        holding_ax.legend(frameon=False, fontsize=9, loc='upper left', ncol=min(4, len(assets)))
        holding_ax.set_ylim(bottom=0)
        holding_ax.margins(y=.30)

        stride = max(1, ceil(len(sessions)/7))
        ticks = sorted(set(range(0, len(sessions), stride)) | {len(sessions)-1})
        for ax in axes:
            ax.set_facecolor('white')
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.spines['bottom'].set_color('#d5deea')
            ax.grid(axis='y', color='#e5ebf2', linewidth=.7)
            ax.set_axisbelow(True)
            ax.tick_params(axis='both', length=0, pad=7, labelsize=9)
            with_initial = ax in (nav_ax, dd_ax) or (ax == holding_ax and daily_accounts is not None)
            ax.set_xlim(-1.5 if with_initial else -.5, len(sessions)-.5)
            ax.set_xticks(([-1]+ticks) if with_initial else ticks, (['t0'] if with_initial else [])+[sessions[j][5:] for j in ticks])
            lo, hi = ax.get_ylim()
            nice = mticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
            ax.set_ylim(nice[0], nice[-1])
            ax.set_yticks(nice)
        figure.text(.075, .043, label(
            '横轴按日对齐；不推断盘中路径。价格未复权，拆分与股息单独入账；买卖点为成交价，柱形为日终资产构成。',
            'Session-aligned axis; lines connect observations, not intraday paths. Markers: simulated fills. Bars: daily holdings.'),
            fontsize=9, color='#607085')
        directory.mkdir(parents=True, exist_ok=True)
        outputs = []
        for extension in ('png', 'svg'):
            target = directory / f'backtest_paths.{extension}'
            kwargs = {'metadata': {'Date': None}} if extension == 'svg' else {}
            figure.savefig(target, dpi=160, facecolor=figure.get_facecolor(), **kwargs)
            outputs.append(target)
        plt.close(figure)
        return outputs
