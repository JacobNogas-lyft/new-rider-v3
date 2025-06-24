WITH

baseline_scores AS (
SELECT join_keys.purchase_session_id,
       max_by(score_for_product_key, event_base.occurred_at) as scores_map
  FROM events.event_purchaseflow_model_executed
 WHERE ds >= '{eval_ds_from}'
   AND ds <= '{ds_to}'
   AND bookkeeping.model_name = '{model_name}'
   AND bookkeeping.model_version = '{model_version}'
   AND score_for_product_key is not NULL
 GROUP BY 1
),

green_rides AS (
  SELECT passenger_lyft_id AS rider_lyft_id,
         DATE_ADD(ds, 1) AS ds,
         MAX(CASE WHEN ride_type='green' THEN rides_28d ELSE 0 END) / MAX(CASE WHEN ride_type='ALL' THEN rides_28d ELSE 1 END) AS proportion_of_green_rides_28d,
         MAX(CASE WHEN ride_type='green' THEN rides_90d ELSE 0 END) / MAX(CASE WHEN ride_type='ALL' THEN rides_90d ELSE 1 END) AS proportion_of_green_rides_90d,
         CAST(MAX(CASE WHEN ride_type='green' THEN booking_28d ELSE 0 END) / MAX(CASE WHEN ride_type='ALL' THEN booking_28d ELSE 1 END) AS DOUBLE) AS proportion_of_green_bookings_28d,
         CAST(MAX(CASE WHEN ride_type='green' THEN booking_90d ELSE 0 END) / MAX(CASE WHEN ride_type='ALL' THEN booking_90d ELSE 1 END) AS DOUBLE) AS proportion_of_green_bookings_90d
    FROM default.fact_passenger_state
   WHERE ds >= DATE_SUB('{ds_from}', 1)
     AND ds <= DATE_SUB('{ds_to}', 1)
     AND passenger_lyft_id IS NOT NULL
   GROUP BY passenger_lyft_id, ds
),

offers AS (
  SELECT purchase_session_id,
         -- Prices used for resolving supply in the pipeline training/evaluation
         MAX(CASE WHEN stable_offer_key='standard' THEN {cost_minor_column} END) AS standard_final_price,
         MAX(CASE WHEN stable_offer_key='courier_reserve_0' THEN {cost_minor_column} END) AS courier_reserve_final_price,
         MAX(CASE WHEN stable_offer_key='plus' THEN {cost_minor_column} END) AS plus_final_price,
         MAX(CASE WHEN stable_offer_key='premium' THEN {cost_minor_column} END) AS premium_final_price,
         MAX(CASE WHEN stable_offer_key='lux' THEN {cost_minor_column} END) AS lux_final_price,
         MAX(CASE WHEN stable_offer_key='luxsuv' THEN {cost_minor_column} END) AS luxsuv_final_price,
         MAX(CASE WHEN stable_offer_key='standard_saver' THEN {cost_minor_column} END) AS standard_saver_final_price,
         MAX(CASE WHEN stable_offer_key='green' THEN {cost_minor_column} END) AS green_final_price,
         MAX(CASE WHEN stable_offer_key='fastpass' THEN {cost_minor_column} END) AS fastpass_final_price,
         MAX(CASE WHEN stable_offer_key='dockless_scooter' THEN {cost_minor_column} END) AS dockless_scooter_final_price,
         MAX(CASE WHEN stable_offer_key='dockable_or_dockless_electric_bike' THEN {cost_minor_column} END) AS dockable_or_dockless_electric_bike_final_price,
         MAX(CASE WHEN stable_offer_key='dock_only_electric_bike' THEN {cost_minor_column} END) AS dock_only_electric_bike_final_price,
         MAX(CASE WHEN stable_offer_key='dock_only_pedal_bike' THEN {cost_minor_column} END) AS dock_only_pedal_bike_final_price,
         -- Availability Caveat for green features
         MAX(CASE WHEN stable_offer_key='green' THEN availability_caveat END) AS green_availability_caveat,
         MAX(CASE WHEN stable_offer_key='pet' THEN availability_caveat END) AS pet_availability_caveat,
         -- ETAs
         MAX(CASE WHEN stable_offer_key='standard' THEN estimated_first_leg_completion_duration_seconds + COALESCE(estimated_first_leg_completion_range_seconds/2.0, 0) END) AS standard_pin_eta,
         MAX(CASE WHEN stable_offer_key='green' THEN estimated_first_leg_completion_duration_seconds + COALESCE(estimated_first_leg_completion_range_seconds/2.0, 0) END) AS green_pin_eta
    FROM events.event_purchaseflow_offer_selector_bundle_placed
   WHERE ds >='{ds_from}'
     AND ds <= '{ds_to}'
     AND is_factual
     AND NOT is_ephemeral -- filter out ephemeral fastpass
   GROUP BY 1
)

  SELECT *
    FROM purchaseflow.finished_rides_models_features
    LEFT JOIN baseline_scores USING (purchase_session_id)
    LEFT JOIN offers USING(purchase_session_id)
    LEFT JOIN green_rides USING(ds, rider_lyft_id)
   WHERE ds >= '{ds_from}'
     AND ds <= '{ds_to}'
     AND {not_null_col} is not null
     AND requested_ride_type IN (
            'standard', 'standard_saver', 'fastpass', 'plus', 'premium', 'lux', 'luxsuv',
            'courier_reserve', 'green', 'pet',
            'dockless_scooter', 'dockable_or_dockless_electric_bike', 'dock_only_electric_bike', 'dock_only_pedal_bike'
         )
   ORDER BY bundle_set_id
   LIMIT {limit}
