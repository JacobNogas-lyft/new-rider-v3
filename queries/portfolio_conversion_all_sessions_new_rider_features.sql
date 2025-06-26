WITH ranked_offers AS (
    SELECT
        purchase_session_id,
        offer_product_key AS atf_mode,
        offer_selector_type,
        is_category_visible_without_user_action,
        CASE
          WHEN category_display_name IN ('Recommended', 'Recomendado','Recommandé', 'For you', 'Para ti', 'Pour vous') THEN TRUE
          ELSE FALSE
        END AS is_atf,
        estimated_route_completion_duration_seconds,
        estimated_travel_time_sec,
        bundle_control_type,
        estimated_route_completion_range_seconds,
        region,
        upfront_cost_minor_units,
        estimated_first_leg_completion_range_seconds,
        CASE
          WHEN is_bundle_preselected = TRUE AND is_offer_preselected_in_bundle = TRUE THEN TRUE
          ELSE FALSE
        END AS is_preselected,
        category_display_name,
        price_quote_id,
        estimated_travel_distance_meters,
        estimated_first_leg_completion_duration_seconds,
        final_user_facing_upfront_cost_minor_units,
        estimated_top_dispatch_candidate_first_leg_completion_duration_seconds,
        RANK() OVER (
            PARTITION BY purchase_session_id, bundle_set_id
            ORDER BY category_rank, bundle_rank_in_category, offer_bundle_index
        ) AS rank
    FROM events.event_purchaseflow_offer_selector_bundle_placed
    WHERE ds BETWEEN '{date_from}' AND '{date_to}'
      AND offer_request_source = 'OFFER_SELECTOR'
      AND is_factual
      AND ranking_procedure LIKE '%new_rider%'
),
aggregated_offer_selector_features AS (
    SELECT
        purchase_session_id,
        offer_selector_type,
        MAX(estimated_travel_time_sec) AS estimated_travel_time_sec, -- same for all ranks
        MAX(price_quote_id) AS price_quote_id, -- same for all ranks
        COUNT(CASE WHEN is_category_visible_without_user_action = 1 THEN 1 END) AS atf_count_category_visible,
        COUNT(CASE WHEN is_atf THEN 1 END) AS atf_count_category_display,
        MAX(CASE WHEN is_preselected THEN atf_mode END) AS preselected_mode,
        MAX(CASE WHEN rank = 1 THEN atf_mode END) AS rank_1,
        MAX(CASE WHEN rank = 2 THEN atf_mode END) AS rank_2,
        MAX(CASE WHEN rank = 3 THEN atf_mode END) AS rank_3,
        MAX(CASE WHEN rank = 4 THEN atf_mode END) AS rank_4,
        MAX(CASE WHEN rank = 5 THEN atf_mode END) AS rank_5,
        MAX(CASE WHEN rank = 6 THEN atf_mode END) AS rank_6,
        MAX(CASE WHEN rank = 7 THEN atf_mode END) AS rank_7,
        MAX(CASE WHEN rank = 8 THEN atf_mode END) AS rank_8,
        MAX(CASE WHEN rank = 9 THEN atf_mode END) AS rank_9,
        MAX(CASE WHEN rank = 10 THEN atf_mode END) AS rank_10,
        MAX(CASE WHEN rank = 11 THEN atf_mode END) AS rank_11,
        MAX(CASE WHEN rank = 12 THEN atf_mode END) AS rank_12,
        MAX(CASE WHEN rank = 1 THEN estimated_route_completion_duration_seconds END) AS rank_1_estimated_route_completion_duration_seconds,
        MAX(CASE WHEN rank = 2 THEN estimated_route_completion_duration_seconds END) AS rank_2_estimated_route_completion_duration_seconds,
        MAX(CASE WHEN rank = 3 THEN estimated_route_completion_duration_seconds END) AS rank_3_estimated_route_completion_duration_seconds,
        MAX(CASE WHEN rank = 4 THEN estimated_route_completion_duration_seconds END) AS rank_4_estimated_route_completion_duration_seconds,
        MAX(CASE WHEN rank = 5 THEN estimated_route_completion_duration_seconds END) AS rank_5_estimated_route_completion_duration_seconds,
        MAX(CASE WHEN rank = 6 THEN estimated_route_completion_duration_seconds END) AS rank_6_estimated_route_completion_duration_seconds,
        MAX(CASE WHEN rank = 7 THEN estimated_route_completion_duration_seconds END) AS rank_7_estimated_route_completion_duration_seconds,
        MAX(CASE WHEN rank = 8 THEN estimated_route_completion_duration_seconds END) AS rank_8_estimated_route_completion_duration_seconds,
        MAX(CASE WHEN rank = 9 THEN estimated_route_completion_duration_seconds END) AS rank_9_estimated_route_completion_duration_seconds,
        MAX(CASE WHEN rank = 10 THEN estimated_route_completion_duration_seconds END) AS rank_10_estimated_route_completion_duration_seconds,
        MAX(CASE WHEN rank = 11 THEN estimated_route_completion_duration_seconds END) AS rank_11_estimated_route_completion_duration_seconds,
        MAX(CASE WHEN rank = 12 THEN estimated_route_completion_duration_seconds END) AS rank_12_estimated_route_completion_duration_seconds,
        MAX(CASE WHEN rank = 1 THEN estimated_route_completion_range_seconds END) AS rank_1_estimated_route_completion_range_seconds,
        MAX(CASE WHEN rank = 2 THEN estimated_route_completion_range_seconds END) AS rank_2_estimated_route_completion_range_seconds,
        MAX(CASE WHEN rank = 3 THEN estimated_route_completion_range_seconds END) AS rank_3_estimated_route_completion_range_seconds,
        MAX(CASE WHEN rank = 4 THEN estimated_route_completion_range_seconds END) AS rank_4_estimated_route_completion_range_seconds,
        MAX(CASE WHEN rank = 5 THEN estimated_route_completion_range_seconds END) AS rank_5_estimated_route_completion_range_seconds,
        MAX(CASE WHEN rank = 6 THEN estimated_route_completion_range_seconds END) AS rank_6_estimated_route_completion_range_seconds,
        MAX(CASE WHEN rank = 7 THEN estimated_route_completion_range_seconds END) AS rank_7_estimated_route_completion_range_seconds,
        MAX(CASE WHEN rank = 8 THEN estimated_route_completion_range_seconds END) AS rank_8_estimated_route_completion_range_seconds,
        MAX(CASE WHEN rank = 9 THEN estimated_route_completion_range_seconds END) AS rank_9_estimated_route_completion_range_seconds,
        MAX(CASE WHEN rank = 10 THEN estimated_route_completion_range_seconds END) AS rank_10_estimated_route_completion_range_seconds,
        MAX(CASE WHEN rank = 11 THEN estimated_route_completion_range_seconds END) AS rank_11_estimated_route_completion_range_seconds,
        MAX(CASE WHEN rank = 12 THEN estimated_route_completion_range_seconds END) AS rank_12_estimated_route_completion_range_seconds,
        MAX(CASE WHEN rank = 1 THEN upfront_cost_minor_units END) AS rank_1_upfront_cost_minor_units,
        MAX(CASE WHEN rank = 2 THEN upfront_cost_minor_units END) AS rank_2_upfront_cost_minor_units,
        MAX(CASE WHEN rank = 3 THEN upfront_cost_minor_units END) AS rank_3_upfront_cost_minor_units,
        MAX(CASE WHEN rank = 4 THEN upfront_cost_minor_units END) AS rank_4_upfront_cost_minor_units,
        MAX(CASE WHEN rank = 5 THEN upfront_cost_minor_units END) AS rank_5_upfront_cost_minor_units,
        MAX(CASE WHEN rank = 6 THEN upfront_cost_minor_units END) AS rank_6_upfront_cost_minor_units,
        MAX(CASE WHEN rank = 7 THEN upfront_cost_minor_units END) AS rank_7_upfront_cost_minor_units,
        MAX(CASE WHEN rank = 8 THEN upfront_cost_minor_units END) AS rank_8_upfront_cost_minor_units,
        MAX(CASE WHEN rank = 9 THEN upfront_cost_minor_units END) AS rank_9_upfront_cost_minor_units,
        MAX(CASE WHEN rank = 10 THEN upfront_cost_minor_units END) AS rank_10_upfront_cost_minor_units,
        MAX(CASE WHEN rank = 11 THEN upfront_cost_minor_units END) AS rank_11_upfront_cost_minor_units,
        MAX(CASE WHEN rank = 12 THEN upfront_cost_minor_units END) AS rank_12_upfront_cost_minor_units,
        MAX(CASE WHEN rank = 1 THEN category_display_name END) AS rank_1_category_display_name,
        MAX(CASE WHEN rank = 2 THEN category_display_name END) AS rank_2_category_display_name,
        MAX(CASE WHEN rank = 3 THEN category_display_name END) AS rank_3_category_display_name,
        MAX(CASE WHEN rank = 4 THEN category_display_name END) AS rank_4_category_display_name,
        MAX(CASE WHEN rank = 5 THEN category_display_name END) AS rank_5_category_display_name,
        MAX(CASE WHEN rank = 6 THEN category_display_name END) AS rank_6_category_display_name,
        MAX(CASE WHEN rank = 7 THEN category_display_name END) AS rank_7_category_display_name,
        MAX(CASE WHEN rank = 8 THEN category_display_name END) AS rank_8_category_display_name,
        MAX(CASE WHEN rank = 9 THEN category_display_name END) AS rank_9_category_display_name,
        MAX(CASE WHEN rank = 10 THEN category_display_name END) AS rank_10_category_display_name,
        MAX(CASE WHEN rank = 11 THEN category_display_name END) AS rank_11_category_display_name,
        MAX(CASE WHEN rank = 12 THEN category_display_name END) AS rank_12_category_display_name,
        MAX(CASE WHEN rank = 1 THEN estimated_travel_distance_meters END) AS estimated_travel_distance_meters, -- same for all ranks
        MAX(CASE WHEN rank = 1 THEN estimated_first_leg_completion_duration_seconds END) AS rank_1_estimated_first_leg_completion_duration_seconds,
        MAX(CASE WHEN rank = 2 THEN estimated_first_leg_completion_duration_seconds END) AS rank_2_estimated_first_leg_completion_duration_seconds,
        MAX(CASE WHEN rank = 3 THEN estimated_first_leg_completion_duration_seconds END) AS rank_3_estimated_first_leg_completion_duration_seconds,
        MAX(CASE WHEN rank = 4 THEN estimated_first_leg_completion_duration_seconds END) AS rank_4_estimated_first_leg_completion_duration_seconds,
        MAX(CASE WHEN rank = 5 THEN estimated_first_leg_completion_duration_seconds END) AS rank_5_estimated_first_leg_completion_duration_seconds,
        MAX(CASE WHEN rank = 6 THEN estimated_first_leg_completion_duration_seconds END) AS rank_6_estimated_first_leg_completion_duration_seconds,
        MAX(CASE WHEN rank = 7 THEN estimated_first_leg_completion_duration_seconds END) AS rank_7_estimated_first_leg_completion_duration_seconds,
        MAX(CASE WHEN rank = 8 THEN estimated_first_leg_completion_duration_seconds END) AS rank_8_estimated_first_leg_completion_duration_seconds,
        MAX(CASE WHEN rank = 9 THEN estimated_first_leg_completion_duration_seconds END) AS rank_9_estimated_first_leg_completion_duration_seconds,
        MAX(CASE WHEN rank = 10 THEN estimated_first_leg_completion_duration_seconds END) AS rank_10_estimated_first_leg_completion_duration_seconds,
        MAX(CASE WHEN rank = 11 THEN estimated_first_leg_completion_duration_seconds END) AS rank_11_estimated_first_leg_completion_duration_seconds,
        MAX(CASE WHEN rank = 12 THEN estimated_first_leg_completion_duration_seconds END) AS rank_12_estimated_first_leg_completion_duration_seconds,
        MAX(CASE WHEN rank = 1 THEN final_user_facing_upfront_cost_minor_units END) AS rank_1_final_user_facing_upfront_cost_minor_units,
        MAX(CASE WHEN rank = 2 THEN final_user_facing_upfront_cost_minor_units END) AS rank_2_final_user_facing_upfront_cost_minor_units,
        MAX(CASE WHEN rank = 3 THEN final_user_facing_upfront_cost_minor_units END) AS rank_3_final_user_facing_upfront_cost_minor_units,
        MAX(CASE WHEN rank = 4 THEN final_user_facing_upfront_cost_minor_units END) AS rank_4_final_user_facing_upfront_cost_minor_units,
        MAX(CASE WHEN rank = 5 THEN final_user_facing_upfront_cost_minor_units END) AS rank_5_final_user_facing_upfront_cost_minor_units,
        MAX(CASE WHEN rank = 6 THEN final_user_facing_upfront_cost_minor_units END) AS rank_6_final_user_facing_upfront_cost_minor_units,
        MAX(CASE WHEN rank = 7 THEN final_user_facing_upfront_cost_minor_units END) AS rank_7_final_user_facing_upfront_cost_minor_units,
        MAX(CASE WHEN rank = 8 THEN final_user_facing_upfront_cost_minor_units END) AS rank_8_final_user_facing_upfront_cost_minor_units,
        MAX(CASE WHEN rank = 9 THEN final_user_facing_upfront_cost_minor_units END) AS rank_9_final_user_facing_upfront_cost_minor_units,
        MAX(CASE WHEN rank = 10 THEN final_user_facing_upfront_cost_minor_units END) AS rank_10_final_user_facing_upfront_cost_minor_units,
        MAX(CASE WHEN rank = 11 THEN final_user_facing_upfront_cost_minor_units END) AS rank_11_final_user_facing_upfront_cost_minor_units,
        MAX(CASE WHEN rank = 12 THEN final_user_facing_upfront_cost_minor_units END) AS rank_12_final_user_facing_upfront_cost_minor_units,
        MAX(CASE WHEN rank = 1 THEN estimated_top_dispatch_candidate_first_leg_completion_duration_seconds END) AS rank_1_estimated_top_dispatch_candidate_first_leg_completion_duration_seconds,
        MAX(CASE WHEN rank = 2 THEN estimated_top_dispatch_candidate_first_leg_completion_duration_seconds END) AS rank_2_estimated_top_dispatch_candidate_first_leg_completion_duration_seconds,
        MAX(CASE WHEN rank = 3 THEN estimated_top_dispatch_candidate_first_leg_completion_duration_seconds END) AS rank_3_estimated_top_dispatch_candidate_first_leg_completion_duration_seconds,
        MAX(CASE WHEN rank = 4 THEN estimated_top_dispatch_candidate_first_leg_completion_duration_seconds END) AS rank_4_estimated_top_dispatch_candidate_first_leg_completion_duration_seconds,
        MAX(CASE WHEN rank = 5 THEN estimated_top_dispatch_candidate_first_leg_completion_duration_seconds END) AS rank_5_estimated_top_dispatch_candidate_first_leg_completion_duration_seconds,
        MAX(CASE WHEN rank = 6 THEN estimated_top_dispatch_candidate_first_leg_completion_duration_seconds END) AS rank_6_estimated_top_dispatch_candidate_first_leg_completion_duration_seconds,
        MAX(CASE WHEN rank = 7 THEN estimated_top_dispatch_candidate_first_leg_completion_duration_seconds END) AS rank_7_estimated_top_dispatch_candidate_first_leg_completion_duration_seconds,
        MAX(CASE WHEN rank = 8 THEN estimated_top_dispatch_candidate_first_leg_completion_duration_seconds END) AS rank_8_estimated_top_dispatch_candidate_first_leg_completion_duration_seconds,
        MAX(CASE WHEN rank = 9 THEN estimated_top_dispatch_candidate_first_leg_completion_duration_seconds END) AS rank_9_estimated_top_dispatch_candidate_first_leg_completion_duration_seconds,
        MAX(CASE WHEN rank = 10 THEN estimated_top_dispatch_candidate_first_leg_completion_duration_seconds END) AS rank_10_estimated_top_dispatch_candidate_first_leg_completion_duration_seconds,
        MAX(CASE WHEN rank = 11 THEN estimated_top_dispatch_candidate_first_leg_completion_duration_seconds END) AS rank_11_estimated_top_dispatch_candidate_first_leg_completion_duration_seconds,
        MAX(CASE WHEN rank = 12 THEN estimated_top_dispatch_candidate_first_leg_completion_duration_seconds END) AS rank_12_estimated_top_dispatch_candidate_first_leg_completion_duration_seconds
    FROM ranked_offers
    GROUP BY purchase_session_id, offer_selector_type
),
modelexec_logs AS (
  SELECT
    ds,
    http_request_id,
    features_json
  FROM events.event_modelexec_predictions_log
  WHERE ds BETWEEN '{date_from}' AND '{date_to}'
    AND model_name = '{model_name}'
    AND version = '{model_version}' -- NOTE: Conversion model 4.0.19 is the compatible version of 4.0.10 in Python 3.10.
),

valid_sessions AS (
  SELECT
    session_id,
    last_purchase_session_id,
    last_http_id,
    dim_user.signup_at signup_at,
    date_diff('day', dim_user.signup_at, date(rs.ds)) AS days_since_signup,
    all_type_total_rides_365d,
    standard_total_rides_365d,
    green_total_rides_365d,
    plus_total_rides_365d,
    premium_total_rides_365d,
    lux_total_rides_365d,
    luxsuv_total_rides_365d,
    standard_saver_total_rides_365d,
    fastpass_total_rides_365d,
    extra_comfort_total_rides_365d,
    courier_reserve_total_rides_365d,
    assisted_ride_total_rides_365d,
    pet_total_rides_365d,
    access_total_rides_365d,
    lyft_disney_total_rides_365d,
    promo_total_rides_365d,
    lyft_disney_access_total_rides_365d,
    rides_lifetime,
    rides_standard_lifetime,
    rides_premium_lifetime,
    rides_plus_lifetime,
    rides_weekday_lifetime,
    rides_weekend_lifetime,
    rides_canceled_lifetime,
    rides_passenger_canceled_lifetime,
    sum_ride_distance_lifetime,
    max_ride_distance_lifetime
  FROM core.rider_sessions rs
  JOIN coco.dim_user dim_user ON rs.rider_lyft_id = dim_user.user_lyft_id
  JOIN features.feature_snapshot_total_rides_last_365d_2 fss on rs.ds = fss.ds and rs.rider_lyft_id = cast(fss.entity_id AS bigint)
  JOIN default.passenger_rides_lifetime lt on rs.rider_lyft_id = lt.user_id and rs.ds = lt.ds
  WHERE is_valid_session
    AND last_purchase_session_id IS NOT NULL
    AND rs.ds BETWEEN '{date_from}' AND '{date_to}'
    AND fss.ds BETWEEN '{date_from}' AND '{date_to}'
),

sessions_with_os_features AS (
  SELECT
    vs.*, 
  --  vs.session_id,
  --  vs.last_http_id,
    os.*
  FROM valid_sessions AS vs
  LEFT JOIN aggregated_offer_selector_features AS os
    ON vs.last_purchase_session_id = os.purchase_session_id
),

sessions_with_os_and_mx_features AS (
  SELECT
    s.*,
    mxl.features_json,
    mxl.ds -- for partitioning
  FROM sessions_with_os_features AS s
  INNER JOIN modelexec_logs AS mxl
    ON s.last_http_id = mxl.http_request_id
)

SELECT
  s.*,
  fr.requested_ride_type,
  CASE
    WHEN fr.rider_session_id IS NOT NULL THEN TRUE
    ELSE FALSE
  END AS is_finished_ride
FROM sessions_with_os_and_mx_features AS s
LEFT JOIN purchaseflow.finished_rides_and_tbs AS fr
  ON s.session_id = fr.rider_session_id
  AND fr.ds BETWEEN '{date_from}' AND '{date_to}'
WHERE s.purchase_session_id IS NOT NULL
  AND RAND({random_seed}) <= {sampling_ratio}
ORDER BY RAND({random_seed})
LIMIT {limit};
