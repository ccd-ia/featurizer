
        select aod.as_of_date, t.*
        from as_of_dates as aod
        cross join lateral (

        with

        
        -- sythetize aggregations and direct features for care_plans
        care_plans_synth as (
        select
        analytics.care_plans.plan_id, analytics.care_plans.effective_at, analytics.care_plans.patient_id, risk_score
        from analytics.care_plans
        
        
        )
        ,
        -- transform care_plans
        care_plans_transform as (
        select
        plan_id, effective_at, patient_id,  abs(risk_score)  as "ABS(care_plans.risk_score)" , risk_score as risk_score
        from care_plans_synth
        )
        ,
        -- sythetize aggregations and direct features for visits
        visits_synth as (
        select
        analytics.visits.visit_id, analytics.visits.visited_at, analytics.visits.patient_id, duration_minutes
        from analytics.visits
        
        
        )
        ,
        -- transform visits
        visits_transform as (
        select
        visit_id, visited_at, patient_id,  abs(duration_minutes)  as "ABS(visits.duration_minutes)" , duration_minutes as duration_minutes
        from visits_synth
        )
        ,
        -- Aggregate for patients
        visits_aggs_for_patients as (
        select
        visits_transform.patient_id,
        count( visit_id ) as "COUNT(visits.visit_id)" ,count( visit_id )  filter (where  daterange((aod.as_of_date - interval 'P1D')::date, aod.as_of_date::date, '[]')  @>  visited_at)  as "COUNT(visits.visit_id|interval=P1D)" ,count( visited_at ) as "COUNT(visits.visited_at)" ,count( visited_at )  filter (where  daterange((aod.as_of_date - interval 'P1D')::date, aod.as_of_date::date, '[]')  @>  visited_at)  as "COUNT(visits.visited_at|interval=P1D)" ,avg( "ABS(visits.duration_minutes)" ) as "MEAN(visits.ABS(visits.duration_minutes))" ,avg( "ABS(visits.duration_minutes)" )  filter (where  daterange((aod.as_of_date - interval 'P1D')::date, aod.as_of_date::date, '[]')  @>  visited_at)  as "MEAN(visits.ABS(visits.duration_minutes)|interval=P1D)" ,avg( duration_minutes ) as "MEAN(visits.duration_minutes)" ,avg( duration_minutes )  filter (where  daterange((aod.as_of_date - interval 'P1D')::date, aod.as_of_date::date, '[]')  @>  visited_at)  as "MEAN(visits.duration_minutes|interval=P1D)" ,percentile_cont(0.5) within group(order by "ABS(visits.duration_minutes)") as "MEDIAN(visits.ABS(visits.duration_minutes))" ,percentile_cont(0.5) within group(order by "ABS(visits.duration_minutes)")  filter (where daterange((aod.as_of_date - interval 'P1D')::date, aod.as_of_date::date, '[]') @> visited_at) as "MEDIAN(visits.ABS(visits.duration_minutes)|interval=P1D)" ,percentile_cont(0.5) within group(order by duration_minutes) as "MEDIAN(visits.duration_minutes)" ,percentile_cont(0.5) within group(order by duration_minutes)  filter (where daterange((aod.as_of_date - interval 'P1D')::date, aod.as_of_date::date, '[]') @> visited_at) as "MEDIAN(visits.duration_minutes|interval=P1D)" ,sum( "ABS(visits.duration_minutes)" ) as "SUM(visits.ABS(visits.duration_minutes))" ,sum( "ABS(visits.duration_minutes)" )  filter (where  daterange((aod.as_of_date - interval 'P1D')::date, aod.as_of_date::date, '[]')  @>  visited_at)  as "SUM(visits.ABS(visits.duration_minutes)|interval=P1D)" ,sum( duration_minutes ) as "SUM(visits.duration_minutes)" ,sum( duration_minutes )  filter (where  daterange((aod.as_of_date - interval 'P1D')::date, aod.as_of_date::date, '[]')  @>  visited_at)  as "SUM(visits.duration_minutes|interval=P1D)" 
        from visits_transform
        where aod.as_of_date >= visited_at
        group by patient_id
        )
        ,
        -- sythetize aggregations and direct features for patients
        patients_synth as (
        select
        analytics.patients.patient_id, analytics.patients.registered_at, "ABS(care_plans.risk_score)", "COUNT(visits.visit_id)", "COUNT(visits.visit_id|interval=P1D)", "COUNT(visits.visited_at)", "COUNT(visits.visited_at|interval=P1D)", "MEAN(visits.ABS(visits.duration_minutes))", "MEAN(visits.ABS(visits.duration_minutes)|interval=P1D)", "MEAN(visits.duration_minutes)", "MEAN(visits.duration_minutes|interval=P1D)", "MEDIAN(visits.ABS(visits.duration_minutes))", "MEDIAN(visits.ABS(visits.duration_minutes)|interval=P1D)", "MEDIAN(visits.duration_minutes)", "MEDIAN(visits.duration_minutes|interval=P1D)", "SUM(visits.ABS(visits.duration_minutes))", "SUM(visits.ABS(visits.duration_minutes)|interval=P1D)", "SUM(visits.duration_minutes)", "SUM(visits.duration_minutes|interval=P1D)", age, risk_score
        from analytics.patients
         left join 
         lateral (
        select
        care_plans_transform."ABS(care_plans.risk_score)" as "ABS(care_plans.risk_score)",
        care_plans_transform.risk_score as risk_score
        from care_plans_transform
        where care_plans_transform.patient_id = analytics.patients.patient_id and care_plans_transform.effective_at <= analytics.patients.registered_at and care_plans_transform.effective_at >= analytics.patients.registered_at - interval 'P14D'
        order by care_plans_transform.effective_at desc
        limit 1
    ) as care_plans_asof_for_patients on true  left join  visits_aggs_for_patients on visits_aggs_for_patients.patient_id = analytics.patients.patient_id 
        )
        ,
        -- transform patients
        patients_transform as (
        select
        patient_id, registered_at,  abs("ABS(care_plans.risk_score)")  as "ABS(care_plans.ABS(care_plans.risk_score))" , "ABS(care_plans.risk_score)" as "ABS(care_plans.risk_score)",  abs("COUNT(visits.visit_id)")  as "ABS(patients.COUNT(visits.visit_id))" ,  abs("COUNT(visits.visit_id|interval=P1D)")  as "ABS(patients.COUNT(visits.visit_id|interval=P1D))" ,  abs("COUNT(visits.visited_at)")  as "ABS(patients.COUNT(visits.visited_at))" ,  abs("COUNT(visits.visited_at|interval=P1D)")  as "ABS(patients.COUNT(visits.visited_at|interval=P1D))" ,  abs("MEAN(visits.ABS(visits.duration_minutes))")  as "ABS(patients.MEAN(visits.ABS(visits.duration_minutes)))" ,  abs("MEAN(visits.ABS(visits.duration_minutes)|interval=P1D)")  as "ABS(patients.MEAN(visits.ABS(visits.duration_minutes)|interval=P1D))" ,  abs("MEAN(visits.duration_minutes)")  as "ABS(patients.MEAN(visits.duration_minutes))" ,  abs("MEAN(visits.duration_minutes|interval=P1D)")  as "ABS(patients.MEAN(visits.duration_minutes|interval=P1D))" ,  abs("MEDIAN(visits.ABS(visits.duration_minutes))")  as "ABS(patients.MEDIAN(visits.ABS(visits.duration_minutes)))" ,  abs("MEDIAN(visits.ABS(visits.duration_minutes)|interval=P1D)")  as "ABS(patients.MEDIAN(visits.ABS(visits.duration_minutes)|interval=P1D))" ,  abs("MEDIAN(visits.duration_minutes)")  as "ABS(patients.MEDIAN(visits.duration_minutes))" ,  abs("MEDIAN(visits.duration_minutes|interval=P1D)")  as "ABS(patients.MEDIAN(visits.duration_minutes|interval=P1D))" ,  abs("SUM(visits.ABS(visits.duration_minutes))")  as "ABS(patients.SUM(visits.ABS(visits.duration_minutes)))" ,  abs("SUM(visits.ABS(visits.duration_minutes)|interval=P1D)")  as "ABS(patients.SUM(visits.ABS(visits.duration_minutes)|interval=P1D))" ,  abs("SUM(visits.duration_minutes)")  as "ABS(patients.SUM(visits.duration_minutes))" ,  abs("SUM(visits.duration_minutes|interval=P1D)")  as "ABS(patients.SUM(visits.duration_minutes|interval=P1D))" ,  abs(age)  as "ABS(patients.age)" , "COUNT(visits.visit_id)" as "COUNT(visits.visit_id)", "COUNT(visits.visit_id|interval=P1D)" as "COUNT(visits.visit_id|interval=P1D)", "COUNT(visits.visited_at)" as "COUNT(visits.visited_at)", "COUNT(visits.visited_at|interval=P1D)" as "COUNT(visits.visited_at|interval=P1D)", "MEAN(visits.ABS(visits.duration_minutes))" as "MEAN(visits.ABS(visits.duration_minutes))", "MEAN(visits.ABS(visits.duration_minutes)|interval=P1D)" as "MEAN(visits.ABS(visits.duration_minutes)|interval=P1D)", "MEAN(visits.duration_minutes)" as "MEAN(visits.duration_minutes)", "MEAN(visits.duration_minutes|interval=P1D)" as "MEAN(visits.duration_minutes|interval=P1D)", "MEDIAN(visits.ABS(visits.duration_minutes))" as "MEDIAN(visits.ABS(visits.duration_minutes))", "MEDIAN(visits.ABS(visits.duration_minutes)|interval=P1D)" as "MEDIAN(visits.ABS(visits.duration_minutes)|interval=P1D)", "MEDIAN(visits.duration_minutes)" as "MEDIAN(visits.duration_minutes)", "MEDIAN(visits.duration_minutes|interval=P1D)" as "MEDIAN(visits.duration_minutes|interval=P1D)", "SUM(visits.ABS(visits.duration_minutes))" as "SUM(visits.ABS(visits.duration_minutes))", "SUM(visits.ABS(visits.duration_minutes)|interval=P1D)" as "SUM(visits.ABS(visits.duration_minutes)|interval=P1D)", "SUM(visits.duration_minutes)" as "SUM(visits.duration_minutes)", "SUM(visits.duration_minutes|interval=P1D)" as "SUM(visits.duration_minutes|interval=P1D)", age as age, risk_score as risk_score
        from patients_synth
        )
        

        select * from patients_transform
        ) as t

        order by aod.as_of_date
        