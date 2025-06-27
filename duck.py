con = duckdb.connect(database=':memory:', read_only=False)
query_sell = """
SELECT distinct
    main.c_depart_int
    ,main.currency
    ,main.start_period
    -----------------------------------------------------------------------------------------
    -- # 4.1 lag n-hour turnover (short-term inertia, P)
    ,sum(aux_1h."Выдано Сумма") as target_prev_1h_sum
    ,sum(aux_3h."Выдано Сумма") as target_prev_3h_sum
    ,sum(aux_6h."Выдано Сумма") as target_prev_6h_sum
    ,sum(aux_12h."Выдано Сумма") as target_prev_12h_sum
    ,sum(aux_24h."Выдано Сумма") as target_prev_24h_sum
      
    /*
    ,sum(case when aux_1h."Выдано Сумма" > 0.1 then 1 else 0 end) as target_prev_1h_cnt
    ,sum(case when aux_3h."Выдано Сумма" > 0.1 then 1 else 0 end) as target_prev_3h_cnt
    ,sum(case when aux_6h."Выдано Сумма" > 0.1 then 1 else 0 end) as target_prev_6h_cnt
    ,sum(case when aux_12h."Выдано Сумма" > 0.1 then 1 else 0 end) as target_prev_12h_cnt
    ,sum(case when aux_24h."Выдано Сумма" > 0.1 then 1 else 0 end) as target_prev_24h_cnt
    */
    -----------------------------------------------------------------------------------------
    
     
FROM operations_sell as main

LEFT JOIN operations_sell as aux_1h 
    on main.c_depart_int = aux_1h.c_depart_int  
    and main.currency = aux_1h.currency 
    and (main.start_period - Interval '1' hour <= aux_1h."Дата и время исполнения" and aux_1h."Дата и время исполнения" < main.start_period )

LEFT JOIN operations_sell as aux_3h 
    on main.c_depart_int = aux_3h.c_depart_int  
    and main.currency = aux_3h.currency 
    and (main.start_period - Interval '3' hour <= aux_3h."Дата и время исполнения" and aux_3h."Дата и время исполнения" < main.start_period )

 
LEFT JOIN operations_sell as aux_6h 
    on main.c_depart_int = aux_6h.c_depart_int  
    and main.currency = aux_6h.currency 
    and (main.start_period - Interval '6' hour <= aux_6h."Дата и время исполнения" and aux_6h."Дата и время исполнения" < main.start_period )
     
LEFT JOIN operations_sell as aux_12h 
    on main.c_depart_int = aux_12h.c_depart_int  
    and main.currency = aux_12h.currency 
    and (main.start_period - Interval '12' hour <= aux_12h."Дата и время исполнения" and aux_12h."Дата и время исполнения" < main.start_period )

LEFT JOIN operations_sell as aux_24h 
    on main.c_depart_int = aux_24h.c_depart_int  
    and main.currency = aux_24h.currency 
    and (main.start_period - Interval '24' hour <= aux_24h."Дата и время исполнения" and aux_24h."Дата и время исполнения" < main.start_period )

-----------------------------------------------------------------------------------------

GROUP BY 1,2,3
"""

operations_sell_autocorr1 = con.execute(query_sell).fetchdf()
con.close()
