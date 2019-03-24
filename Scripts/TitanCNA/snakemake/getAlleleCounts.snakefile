configfile: "config/config.yaml"
configfile: "config/samples.yaml"

CHRS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,'X']

rule tumCounts:
	input: 
		expand("results/titan/tumCounts/{tumor}/{tumor}.tumCounts.chr{chr}.txt", tumor=config["pairings"], chr=CHRS),		
		expand("results/titan/tumCounts/{tumor}.tumCounts.txt", tumor=config["pairings"])
	
rule catAlleleCountFiles:
	input:
		expand("results/titan/tumCounts/{{tumor}}/{{tumor}}.tumCounts.chr{chr}.txt", chr=CHRS)
	output:
		"results/titan/tumCounts/{tumor}.tumCounts.txt"
	log:
		"logs/titan/tumCounts/{tumor}/{tumor}.cat.log"
	shell:
		"cat {input} | grep -v Chr > {output} 2> {log}"






