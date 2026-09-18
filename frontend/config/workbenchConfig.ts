import { Activity, BarChart3, Boxes, Library, ShieldCheck, Table2, Target, TrendingUp } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import type { BoardId } from '../types'

export type ComponentId = 'foundation' | 'factorLab'
export type WidgetSize = { columns: number; rows: number }
export type WorkbenchTab = 'strategy' | 'factorReturns' | 'factorSets' | 'scoring' | 'optimizer' | 'backtest' | 'simulation'
export type WorkbenchTabStatus = 'ready' | 'planned'

export type ToolboxTabConfig = {
  id: WorkbenchTab
  label: string
  icon: LucideIcon
  note: string
  status: WorkbenchTabStatus
}

export type ComponentCatalogEntry = {
  id: ComponentId
  title: string
  subtitle: string
  icon: LucideIcon
  size: 'wide' | 'normal'
}

export const boardTabs: Array<{ id: BoardId; label: string }> = [
  { id: 'MAIN', label: '沪深主板' },
  { id: 'CHINEXT', label: '创业板' },
  { id: 'STAR', label: '科创板' },
  { id: 'BSE', label: '北交所' },
]

export const componentCatalog: ComponentCatalogEntry[] = [
  {
    id: 'foundation',
    title: '基础数据边界',
    subtitle: '展示当前研究范围与数据覆盖情况',
    icon: ShieldCheck,
    size: 'wide',
  },
  {
    id: 'factorLab',
    title: '因子注册表',
    subtitle: '查看已登记因子的定义与构造信息',
    icon: Library,
    size: 'wide',
  },
]

export const defaultWorkbenchComponents: ComponentId[] = ['foundation', 'factorLab']

export const defaultWidgetSizes: Record<ComponentId, WidgetSize> = {
  foundation: { columns: 12, rows: 14 },
  factorLab: { columns: 12, rows: 20 },
}

export const workbenchTabMeta: Record<WorkbenchTab, { label: string; icon: LucideIcon }> = {
  strategy: { label: '因子实验室', icon: Library },
  factorReturns: { label: '因子收益', icon: Activity },
  factorSets: { label: '因子集合', icon: Boxes },
  scoring: { label: '打分排名', icon: Table2 },
  optimizer: { label: '优化器·对照', icon: Target },
  backtest: { label: '回测实验', icon: BarChart3 },
  simulation: { label: '模拟盘', icon: TrendingUp },
}

export const workbenchTabOrder: WorkbenchTab[] = ['strategy', 'factorReturns', 'factorSets', 'scoring', 'optimizer', 'backtest', 'simulation']
export const defaultOpenTabs: WorkbenchTab[] = ['strategy', 'scoring']

export const workbenchTools: ToolboxTabConfig[] = [
  { id: 'strategy', icon: Library, label: '因子实验室', note: '构造、单因子验证、关系分析', status: 'ready' },
  { id: 'factorReturns', icon: Activity, label: '横截面与因子收益', note: '回归收益、协方差、特异收益', status: 'planned' },
  { id: 'factorSets', icon: Boxes, label: '因子集合审批', note: '等待新的集合契约', status: 'planned' },
  { id: 'scoring', icon: Table2, label: '评分排名', note: '等待新的显式评分契约', status: 'planned' },
  { id: 'optimizer', icon: Target, label: '优化器·对照', note: '消费净Alpha与风险模型，提供第二意见', status: 'planned' },
  { id: 'backtest', icon: BarChart3, label: '评估框架', note: '负对照、IC 推断与样本边界', status: 'ready' },
  { id: 'simulation', icon: TrendingUp, label: '模拟盘', note: '仿真交易与盘中风险控制', status: 'planned' },
]

export const factorFamilyOptions = [
  { value: 'alpha' as const, label: 'Alpha' },
  { value: 'risk' as const, label: '风险' },
]

export const factorRoleLabel: Record<string, string> = {
  country: 'Country',
  industry: '行业',
  board: '板块',
  membership: '成员',
  style_risk: '风格风险',
  alpha_candidate: 'Alpha候选',
  descriptor: '描述变量',
}

export const workbenchCopy = {
  topbar: {
    homeLabel: '工作台',
    searchHint: '搜索研究工作区',
    pipelineButton: '数据管道',
    allRefresh: '全部刷新',
    addTab: '添加工作区',
    addComponent: '添加组件',
    removeTabAriaPrefix: '删除',
    removeTabAriaSuffix: '工作区',
  },
  sidebar: {
    title: '研究工具',
    boardAriaLabel: '上市板块',
    boardSuffix: '只',
    toolboxTitle: '研究工具',
    toolboxOpenTitle: '打开研究工具',
    toolboxCloseTitle: '关闭研究工具箱',
    closeButton: '关闭组件库',
    widgetHomeLabel: '工作台',
  },
  boardTabs: {
    titlePrefix: 'FACTOR LAB · ',
  },
  workbenchStates: {
    strategy: {
      title: '因子实验室',
      subtitle: '从基础数据开始浏览并登记研究因子。',
      warningTitle: '等待选择因子',
      warningNote: '选择或搜索因子后查看详细信息',
    },
    foundation: {
      heading: '基础数据边界',
      noData: {
        matrixLabel: '矩阵',
        matrixTitle: '市场覆盖范围',
        matrixStatus: '已生成',
        qualityLabel: '质量',
        qualityTitle: '基础质量闸门',
        qualityStatusText: '动态读取',
        noFactorTag: '当前工作区展示基础数据，不包含因子、评分或组合结果。',
        sampleCountSuffix: ' 条样本',
        blockersSuffix: ' 个待处理项',
      },
    },
    factorLab: {
      heading: '因子注册表',
      loadingHint: '输入关键词检索注册表；注册表不会加载暴露值或诊断结果。',
      emptyText: '未找到匹配因子。',
      tableHeaders: ['因子 / 构造', '角色', '处理管线', '状态', '假设'],
      orthogonalizePrefix: '结构列 / 正交根',
      afterLabel: 'after: ',
      forbidDeploySuffix: ' · 禁止部署',
      variantPrefix: 'variants ',
      sourceHint: '同一物理注册表 · 按 family 分视图',
      searchPlaceholder: '搜索名称、ID、公式或族系',
      bottom: '研究态 · 未发送 N×K 因子矩阵',
      placeholderSuffix: {
        strategy: '策略',
        waitingSearch: '等待检索',
        searchSummaryPrefix: '找到',
        searchSummarySuffix: '个 · 最多显示',
      },
    },
  },
  toolbar: {
    boardTitle: '计算板块',
    workflowTitle: '研究工作流',
    quickTitle: '快捷操作',
    quickActions: {
      openPipeline: '打开数据管道',
      refreshRegistry: '刷新注册表',
      openRegistry: '打开因子注册表',
      addWidget: '添加研究组件',
    },
    statusText: '任何新计算都必须显式注册版本、参数、输入和输出。',
  },
  widget: {
    dragLabel: '按住拖动组件',
    resizeLabel: '按住拖动调整大小',
    dragAriaPrefix: '拖动',
    refreshAriaPrefix: '刷新',
    deleteAriaPrefix: '删除',
    resizeAriaPrefix: '调整',
    resizeAriaSuffix: '大小',
  },
  shell: {
    closeMenu: '关闭菜单',
    workspacePipeline: '数据管道',
    workspaceResearch: '因子研究',
    settings: '设置',
    search: '搜索',
    localEnvironment: '本地研究环境',
    localState: '真实本地数据',
    openMenu: '打开菜单',
    searchAria: '搜索',
  },
    locked: {
    draftState: {
      ready: '可用',
      planned: '规划',
    },
    noWorkspace: {
      title: '暂无工作区',
      description: '点击顶部加号恢复已关闭的研究工作区。',
      action: '添加工作区',
    },
    emptyWidgets: {
      title: '工作区暂无组件',
      description: '使用右下角按钮添加研究组件。',
    },
    componentLibrary: {
      previewBadge: 'RESEARCH',
      title: '添加研究组件',
      label: 'COMPONENT LIBRARY',
      closeButton: '关闭组件库',
      searchPlaceholder: '搜索研究组件',
      sectionTitle: '研究组件',
      completeText: '已添加',
      addText: '点击添加',
      footerLabel: '已添加 {count} 个组件 · 可拖动排序和缩放',
      doneButton: '完成',
    },
    factorReturns: {
      eyebrow: 'FACTOR RETURNS',
    },
    quality: {
      title: '因子收益层等待重新设计',
      description: '旧横截面回归实现已清除；新模型需显式注册后开放。',
    },
    factorSets: {
      eyebrow: 'FACTOR SETS',
      title: '因子集合层等待重新设计',
      description: '当前没有已验证因子，也没有隐式默认集合。',
    },
    scoring: {
      eyebrow: 'SCORING · {board}',
      title: '评分层等待重新设计',
      description: '旧评分公式与结果已清除；新评分契约登记后再开放。',
    },
    optimizer: {
      eyebrow: 'OPTIMIZER · COMPARATOR',
      title: '优化器等待真实输入',
      description: '等待独立研究层与风险层重新提供已验证输入。',
    },
    backtest: {
      eyebrowPrefix: 'EVALUATION · ',
      title: '估计区间评估框架',
      description: '先验证读数可信，再比较信号、频率与风险模型。',
    },
    simulation: {
      eyebrow: 'PAPER TRADING',
      title: '模拟盘尚未连接',
      description: '等待真实行情、组合和统一执行规则。',
    },
  },
  factorLab: {
    loadingHint: '输入关键词检索注册表；注册表不会加载暴露值或诊断结果。',
    emptyText: '未找到匹配因子。',
    tableHeaders: ['因子 / 构造', '角色', '处理管线', '状态', '假设'],
    orthogonalizePrefix: '结构列 / 正交根',
    afterLabel: 'after: ',
    forbidDeploySuffix: ' · 禁止部署',
    variantPrefix: 'variants ',
    sourceHint: '同一物理注册表 · 按 family 分视图',
    searchPlaceholder: '搜索名称、ID、公式或族系',
    bottom: '研究态 · 未发送 N×K 因子矩阵',
    placeholderSuffix: {
      strategy: '策略',
      waitingSearch: '等待检索',
      searchSummaryPrefix: '找到',
      searchSummarySuffix: '个 · 最多显示',
    },
  },
  foundation: {
    heading: '基础数据边界',
    noData: {
      matrixLabel: '矩阵',
      matrixTitle: '市场覆盖范围',
      matrixStatus: '已生成',
      qualityLabel: '质量',
      qualityTitle: '基础质量闸门',
      qualityStatusText: '动态读取',
      noFactorTag: '当前工作区展示基础数据，不包含因子、评分或组合结果。',
      sampleCountSuffix: ' 条样本',
      blockersSuffix: ' 个待处理项',
    },
  },
  componentLibrary: {
    previewBadge: 'RESEARCH',
    title: '添加研究组件',
    label: 'COMPONENT LIBRARY',
    closeButton: '关闭组件库',
    searchPlaceholder: '搜索研究组件',
    sectionTitle: '研究组件',
    completeText: '已添加',
    addText: '点击添加',
    footerLabel: '已添加 {count} 个组件 · 可拖动排序和缩放',
    doneButton: '完成',
  },
  errors: {
    factorRegistryLoadFailed: '因子注册表读取失败：',
  },
} as const

export const draftCopy = {
  breadcrumb: 'FACTOR MATRIX',
  brandTitle: {
    pipeline: 'Data Pipeline',
    research: 'Factor Matrix',
  },
  brandDesc: {
    pipeline: 'A-SHARE OPERATIONS',
    research: 'A-SHARE RESEARCH',
  },
  boardEnvText: {
    pipeline: 'Silver 质量通过',
    research: 'Gold 市场矩阵已生成',
  },
  avatarText: '研',
  shell: {
    localEnvironmentLabel: '本地研究环境',
    closeMenu: '关闭菜单',
    settings: '设置',
    searchAria: '搜索',
    openMenu: '打开菜单',
    workspacePipeline: '数据管道',
    workspaceResearch: '因子研究',
    localState: '真实本地数据',
  },
} as const
