--https://app.mode.com/lyft/reports/5f82c0816fac/details/queries/f97cd879c0e7
--higherlimit
WITH POD_offers AS (
  SELECT
    join_keys.region,
    join_keys.purchase_session_id purchase_session_id,
    a.ride_id,
    a.offer_id,
    dispatch_signals.match_efficiency_v2.expected_match_efficiency AS oracle_me,
    dispatch_signals.pickup_estimate.expected_pickup_window.duration AS pin_eta,
    dispatch_signals.dropoff_estimate.expected_dropoff_window.duration AS pin_etd,
    CASE
      WHEN ranking_procedure LIKE '%new_rider%' THEN TRUE
      ELSE FALSE
    END is_new_rider
  FROM
    event_constraints_applied a
    JOIN event_purchaseflow_offer_selector_bundle_placed b ON a.join_keys.purchase_session_id = b.purchase_session_id
    AND a.ds = b.ds
    AND a.stable_offer_key = b.stable_offer_key
  WHERE
    a.ds BETWEEN '{{start_date_full_nr}}'
    AND '{{end_date}}'
    AND b.ds BETWEEN '{{start_date_full_nr}}'
    AND '{{end_date}}'
    AND a.stable_offer_key IN (
      'courier_reserve',
      'premium',
      'fastpass',
      'green',
      'lux',
      'luxsuv',
      'pet',
      'plus',
      'standard',
      'standard_saver',
      'extra_comfort'
    )
),
requested_rides AS (
  SELECT
    a.*,
    a.ride_id,
    b.requested_to_dropped_off_seconds,
    b.analytical_ride_type,
    date_trunc('week', date(ds)) AS week_start
  FROM
    POD_offers a
    JOIN coco.fact_rides b ON a.ride_id = b.ride_id
  WHERE
    b.ds BETWEEN '{{start_date_full_nr}}'
    AND '{{end_date}}'
    AND requested_to_dropped_off_seconds IS NOT NULL
    AND analytical_ride_type IN (
      'courier_reserve',
      'premium',
      'fastpass',
      'green',
      'lux',
      'luxsuv',
      'pet',
      'plus',
      'standard',
      'standard_saver',
      'extra_comfort'
    )
    AND destination_pin.airport_code IS NOT NULL
)
SELECT
  week_start,
  avg(
    CASE
      WHEN analytical_ride_type = 'standard' THEN requested_to_dropped_off_seconds - pin_ETD
      ELSE NULL
    END
  ) avg_standard_diff_requested_to_dropped_off_seconds_pin_etd,
  avg(
    CASE
      WHEN analytical_ride_type = 'standard_saver' THEN requested_to_dropped_off_seconds - pin_ETD
      ELSE NULL
    END
  ) avg_ws_diff_requested_to_dropped_off_seconds_pin_etd,
  avg(
    CASE
      WHEN analytical_ride_type = 'standard'
      AND is_new_rider = TRUE THEN requested_to_dropped_off_seconds - pin_ETD
      ELSE NULL
    END
  ) avg_new_rider_standard_diff_requested_to_dropped_off_seconds_pin_etd,
  avg(
    CASE
      WHEN analytical_ride_type = 'standard_saver'
      AND is_new_rider = TRUE THEN requested_to_dropped_off_seconds - pin_ETD
      ELSE NULL
    END
  ) avg_new_rider_ws_diff_requested_to_dropped_off_seconds_pin_etd
FROM
  requested_rides
GROUP BY
  1
 --At ride level, what is the difference between requested and dropped off time?
  
  
   -- mode_rollup AS (
  --   SELECT
  --     is_new_rider,
  --     analytical_ride_type,
  --     week_start,
  --     avg(
  --       CASE
  --         WHEN requested_to_dropped_off_seconds > pin_ETD + 180 THEN 1 --WHEN requested_to_dropped_off_seconds < pin_ETD - 120 THEN 'early'
  --         ELSE 0
  --       END
  --     ) AS ETD_reliability_avg,
  --     AVG(requested_to_dropped_off_seconds - pin_ETD) avg_diff_requested_to_dropped_off_seconds_pin_etd
  --   FROM
  --     requested_rides
  --   GROUP BY
  --     1,
  --     2,
  --     3
  -- )
  -- nr_week_rollup as (
  -- SELECT
  --   is_new_rider,
  --   week_start,
  --   CASE
  --     WHEN analytical_ride_type = 'standard' THEN avg_diff_requested_to_dropped_off_seconds_pin_etd
  --     ELSE NULL
  --   END standard_avg_diff_requested_to_dropped_off_seconds_pin_etd,
  --   CASE
  --     WHEN analytical_ride_type = 'standard_saver' THEN avg_diff_requested_to_dropped_off_seconds_pin_etd
  --     ELSE NULL
  --   END ws_avg_diff_requested_to_dropped_off_seconds_pin_etd
  -- FROM
  --   mode_rollup)
  --   select 
  --     week_start,
  --     case when is_new_rider standard_avg_diff_requested_to_dropped_off_seconds_pin_etd
  --     ws_avg_diff_requested_to_dropped_off_seconds_pin_etd