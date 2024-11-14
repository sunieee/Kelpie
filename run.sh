
python explain.py --dataset WN18 --model complex --baseline k1_abstract --perspective double > out/complex_WN18/explain.log
python process.py --dataset WN18 --model complex --alpha 0.05 > out/complex_WN18/process.log

python verify.py --dataset WN18 --model complex --metric score --topN 4 > out/complex_WN18/verify_score4.log
python verify.py --dataset WN18 --model complex --metric score --topN 4 --filter head > out/complex_WN18/verify_score_h4.log