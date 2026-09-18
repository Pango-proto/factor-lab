export type BoardId = 'MAIN' | 'CHINEXT' | 'STAR' | 'BSE'

export type Workspace = 'pipeline' | 'research'
export type Page = 'overview' | 'quality' | 'sources'

export type Snapshot = {
  as_of_date: string; matrix_id: string
  counts: { listed: number; eligible_market_matrix: number; eligible_industry_model: number; feature_values: number; populated_feature_values: number; feature_count: number; market_feature_count: number; financial_feature_count: number; by_board?: Record<BoardId, { listed: number; eligible_market_matrix: number }> }
  exclusions: Record<string, number>
  features: Array<{ id: string; label: string; format: string }>
  sample: Array<Record<string, string | number | null>>
  stages: Array<{ id: string; label: string; status: 'complete' | 'blocked' }>
  quality_checks: { duplicate_assets: number; future_available_values: number; missing_market_cap: number }
}

export type PipelineSummary = {
  status: string; as_of_date: string | null
  history: { min_trade_date?: string; max_trade_date?: string; price_rows?: number; trading_days?: number; assets?: number }
  tables: Array<{ id: string; label: string; status: 'complete' | 'missing'; rows: number }>
}

export type FactorCatalogResult = {
  status: string; query: string; total: number; page: number; page_size: number
  rows: Array<{
    factor_id: string; version: number; label: string; family: 'alpha' | 'risk'
    role: 'country' | 'industry' | 'board' | 'membership' | 'style_risk' | 'alpha_candidate' | 'descriptor'
    feature_key: string; formula: string; description: string
    source_tables: string[]; source_fields: string[]; pit_key: string
    winsorize_method: string | null; standardize: string | null; weight_scheme: string | null
    neutralize_against: string | string[] | null; orthogonalize_after: string[]
    coverage_min: number | null; variant_count: number; family_root_id: string; risk_set_version: number | null
    research_status: string; deployment_status: string
  }>
}

export type IcSummary = {
  observations: number
  mean: number
  median: number
  standard_deviation: number
  ic_ir: number
  newey_west_lag: number
  newey_west_standard_error: number
  newey_west_t: number
}

export type EvaluationFramework = {
  status: 'ready' | 'registered_not_run' | 'superseded' | 'stale' | 'blocked'
  framework_id: string
  config_sha256: string
  review?: null | {
    execution_status: string
    implementation_status: string
    temporal_validation_status: string
    research_status: string
    required_gates: Record<string, boolean>
    blocking_reasons: string[]
    structured_diagnostics: Record<string, { disposition: string; legacy_zero_interval_passed: boolean }>
  }
  excluded_run: null | {
    status: string | null
    reason: string
    superseded_by: string | null
    supersession_reason: string | null
    source: string
  }
  contract: {
    status: string
    component_declaration: {
      component: string
      protects: string
      failure_mode: string
      downstream_consequence: string
    }
    sample: {
      development_start: string
      holdout_start: string
      holdout_status: string
      horizon_unit: string
      reported_horizons: number[]
      primary_horizon: number
      embargo_trading_days: number
      purge_gap_trading_days: number
    }
    signal: { signal_id: string; definition: string; role: string; risk_neutralization: string[] }
    evaluation_channel: { variant_dispositions?: Record<string, 'gate' | 'sensitivity_only' | 'diagnostic' | 'structured_diagnostic'> }
    construction_gates: {
      T3_next_day_execution_filter: {
        status: string
        unblocked_by?: string
        execution_date_rule?: string
        evidence?: string
      }
    }
    output_contract: { implemented: string[]; next: string[] }
    holdout_capacity: { as_of?: string; approximate_trading_days: number; intended_use: string; cannot_confirm: string }
  }
  latest_run: null | {
    status: string
    scope: { start: string; end: string; safe_feature_end: string; primary_horizon: number }
    data_quality: { valid_daily_cross_sections: number; median_cross_section: number; total_seal_events: number }
    ic: { by_horizon: Record<string, IcSummary>; decay_curve: Array<IcSummary & { horizon_trading_days: number }> }
    negative_controls: {
      passed: boolean
      within_date_cross_sectional_return_shuffle?: Record<string, IcSummary & { passed: boolean; absolute_mean: number; absolute_mean_limit: number }>
      whole_cross_section_time_axis_shuffle?: Record<string, IcSummary & { passed: boolean; absolute_mean: number; absolute_mean_limit: number }>
      label_window_shift_peak_at_zero?: { passed: boolean; peak_offsets_trading_days: number[] }
      raw_signal_shift_peak_at_zero?: { passed: boolean; peak_offsets_trading_days: number[] }
      within_asset_time_shuffle_100_repetition_asset_demeaned_null?: {
        passed: boolean; lower: number; upper: number; repetitions: number
      }
    }
    backtest: null | { mean_turnover?: number; cost_sensitivity_bps_per_turnover: Record<string, { cumulative_net_return: number }> }
  }
  boundary: { daily_ic_sent: boolean; security_returns_sent: boolean; factor_matrix_sent: boolean; holdout_opened: boolean }
}
