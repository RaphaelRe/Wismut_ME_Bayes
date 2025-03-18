echo "Fitting simulation study..."

jupyter nbconvert --execute --to notebook --output out_b3_naive --output-dir ./out run_model_b3-m5-naive.ipynb
jupyter nbconvert --execute --to notebook --output out_b3 --output-dir ./out run_model_b3-m5.ipynb
jupyter nbconvert --execute --to notebook --output out_b6_naive --output-dir ./out run_model_b6-m5-naive.ipynb
jupyter nbconvert --execute --to notebook --output out_b6 --output-dir ./out run_model_b6-m5.ipynb
jupyter nbconvert --execute --to notebook --output out_misspec1 --output-dir ./out run_model_b3-m5-missspec1.ipynb
jupyter nbconvert --execute --to notebook --output out_misspec2 --output-dir ./out run_model_b3-m5-missspec2.ipynb

echo "Model fitting is done! Generating results..."


jupyter nbconvert --execute --to notebook --output results_simulation_study --output-dir ./out analyze_simulation_study.ipynb
echo "Done simulation study finished!"
