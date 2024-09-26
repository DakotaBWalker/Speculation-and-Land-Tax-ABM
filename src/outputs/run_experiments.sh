

echo 'running lu scenarios'
for RI in {1..20}
do
    for LUS in 'baseline' 'sprawl' 'density'
    do
        for TS in 'SQ' 'LVT' 'ELVT'
        do
            EN='LU_scenarios'
            echo $RI
            echo $LUS
            echo $TS
            SNR=1
            EBD=100
            LVTR=0.05
            python main.py --experiment_name $EN --run_id $RI --LU_scenario $LUS --tax_scheme $TS --speculator_nr $SNR --eco_burden_denom $EBD --lvt_rate $LVTR
        done
    done
done

echo 'running speculator sweep'
for RI in {1..20}
do
    for SNR in 0 1 2 3 4
    do
        for TS in 'SQ' 'LVT' 'ELVT'
        do
            EN='speculator_sweep'
            echo $RI
            LUS='baseline'
            echo $TS
            echo $SNR
            EBD=100
            LVTR=0.05
            python main.py --experiment_name $EN --run_id $RI --LU_scenario $LUS --tax_scheme $TS --speculator_nr $SNR --eco_burden_denom $EBD --lvt_rate $LVTR
        done
    done
done

echo 'running eco burden sweep'
for RI in {1..20}
do
    for EBD in 140 110 80 50 20
    do
        for TS in 'LVT' 'ELVT'
        do
            EN='eco_burden_sweep'
            echo $RI
            LUS='baseline'
            echo $TS
            SNR=1
            echo $EBD
            LVTR=0.05
            python main.py --experiment_name $EN --run_id $RI --LU_scenario $LUS --tax_scheme $TS --speculator_nr $SNR --eco_burden_denom $EBD --lvt_rate $LVTR
        done
    done
done

echo 'running lvt rate sweep'
for RI in {1..20}
do
    for LVTR in 0.05 0.1 0.15 0.2 0.25
    do
        for TS in 'LVT' 'ELVT'
        do
            EN='lvt_rate_sweep'
            echo $RI
            LUS='baseline'
            echo $TS
            SNR=1
            EBD=100
            echo $LVTR
            python main.py --experiment_name $EN --run_id $RI --LU_scenario $LUS --tax_scheme $TS --speculator_nr $SNR --eco_burden_denom $EBD --lvt_rate $LVTR
        done
    done
done

