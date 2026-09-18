import { Database, LayoutDashboard, ShieldCheck } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import type { Page } from '../types'

export type PipelineNavItem = {
  id: Page
  label: string
  icon: LucideIcon
}

export const pipelineNav: PipelineNavItem[] = [
  { id: 'overview', label: '管道总览', icon: LayoutDashboard },
  { id: 'quality', label: '质量报告', icon: ShieldCheck },
  { id: 'sources', label: '数据目录', icon: Database },
]

export const pipelineUiCopy = {
  breadcrumb: {
    pipeline: '数据管道',
    research: '因子研究',
  },
  overview: {
    eyebrow: 'DATA OPERATIONS',
    title: '数据管道控制台',
    description: '只负责抓取、标准化、质量和可追溯性，不展示模拟因子结果。',
    marketBadge: '市场层质量通过',
    card: {
      tradingDays: {
        title: '连续交易日',
        valueMissing: '读取中',
      },
      marketQuote: {
        title: 'A 股行情',
        unit: ' 个历史证券代码',
        valueMissing: '读取中',
      },
      gating: {
        title: '最新矩阵质量闸门',
        missing: '读取中',
        suffix: ' 项阻断',
        fallbackDate: '等待快照',
      },
      boundary: {
        title: '管道职责边界',
        badge: '凭据仅在后端钥匙串读取',
      },
      pipelineNode: {
        subtitle: '研究任务只读取内部标准表，不直接调用 Tushare。',
        nodes: [
          ['Bronze', '供应商原始响应'],
          ['Silver', '行情、估值、收益'],
          ['Quality', '最新日 + 全历史'],
          ['Gold', '独立生成研究矩阵'],
        ] as const,
        detailSuffix: '· 动态读取Gold质量检查',
        boundarySuffix: '与研究域解耦',
      },
    },
  },
  quality: {
    subtitle: 'QUALITY GATE',
    title: '市场矩阵质量报告',
    intro: '当前页面读取最新Gold快照；完整日行情报告保存在本地质量目录。',
    checksTitle: '质量闸门统计',
    checks: [
      '矩阵资产主键重复',
      '截止日后才可用的特征值',
      '合格矩阵缺少有效市值',
    ],
    checkTitles: [
      '矩阵资产主键重复',
      '截止日后才可用的特征值',
      '合格矩阵缺少有效市值',
    ] as const,
    noteTitle: '质量口径说明',
    note: '安全排除项与阻断项分开统计；行业L2覆盖和因子缺失请在因子策略的“质量与前视检查”组件查看。',
    statusCopy: {
      pass: '通过',
      fail: '失败',
      waiting: '等待快照',
      blockerSuffix: ' 项阻断',
      issueSuffix: ' 个问题',
    },
  },
  sources: {
    title: '标准数据目录',
    intro: '只展示输入事实表的后端摘要，不混入策略结果。',
    catalogTitle: 'SILVER CATALOG',
    header: ['数据域', '内部表', '覆盖', '状态'],
    rowUnit: '行',
    badge: {
      complete: '完成',
      missing: '缺失',
    },
  },
  loadingErrorPrefix: '数据接口读取失败：',
} as const
