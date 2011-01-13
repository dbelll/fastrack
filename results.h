/*
 *  results.h
 *  fastrack
 *
 *  Created by Dwight Bell on 1/1/11.
 *  Copyright 2011 dbelll. All rights reserved.
 *
 */

// calculate the winning percentage based on games, wins and losses in WON_LOSS structure
float winpct(WON_LOSS wl)
{
	return 0.5f * ((float)(wl.wins + (wl.games - wl.losses))/(float)wl.games);
}

// comparison function to sort by won/loss percentage
int wl_compare(const void *p1, const void *p2)
{
	const WON_LOSS *wl1 = (const WON_LOSS *)p1;
	const WON_LOSS *wl2 = (const WON_LOSS *)p2;
	float score1 = (wl1->wins + (wl1->games - wl1->losses)) / (float)wl1->games;
	float score2 = (wl2->wins + (wl2->games - wl2->losses)) / (float)wl2->games;
	int result = 0;
	if (score1 > score2) result = -1;
	if (score1 < score2) result = 1;
	//	printf("(%d-%d-%d  %5.3f) vs (%d-%d-%d  %5.3f) comparison is %d\n", wl1->wins, wl1->losses, wl1->games - wl1->wins - wl1->losses, score1, wl1->wins, wl2->losses, wl2->games - wl2->wins - wl2->losses, score2, result);
	return result; 
}

// comparison function to sort by agent
int wl_byagent(const void *p1, const void *p2)
{
	const WON_LOSS *wl1 = (const WON_LOSS *)p1;
	const WON_LOSS *wl2 = (const WON_LOSS *)p2;
	int result = 0;
	if (wl1->agent < wl2->agent) result = -1;
	if (wl1->agent > wl2->agent) result = 1;
	return result; 
}

// Allocate a RESULTS struture to hold results for p.num_sessions
RESULTS *newResults(PARAMS p)
{
	RESULTS *results = (RESULTS *)malloc(sizeof(RESULTS));
	results->p = p;
	results->standings = (WON_LOSS *)malloc(p.num_sessions * p.num_agents * sizeof(WON_LOSS));
	results->vsChamp = (WON_LOSS *)malloc(p.num_sessions * p.num_agents * sizeof(WON_LOSS));
	return results;
}

// Allocate a RESULTS struture to hold results for p.num_sessions on the device
RESULTS *newResultsGPU(PARAMS p)
{
	// build the structure on host, with pointers to device memory areas for standings and vsChamp
	RESULTS *results = (RESULTS *)malloc(sizeof(RESULTS));
	results->p = p;
	CUDA_SAFE_CALL(cudaMalloc(&results->standings, p.num_sessions * p.num_agents * sizeof(WON_LOSS)));
	CUDA_SAFE_CALL(cudaMalloc(&results->vsChamp, p.num_sessions * p.num_agents * sizeof(WON_LOSS)));
	
	// set newly allocated memory to all zeroes
	CUDA_SAFE_CALL(cudaMemset(results->standings, 0, p.num_sessions * p.num_agents * sizeof(WON_LOSS)));
	CUDA_SAFE_CALL(cudaMemset(results->vsChamp, 0, p.num_sessions * p.num_agents * sizeof(WON_LOSS)));
	return results;
}


void freeResults(RESULTS *r)
{
	if (r) {
		if (r->standings) free(r->standings);
		if (r->vsChamp) free(r->vsChamp);
		free(r);
	}
}

void freeResultsGPU(RESULTS *r)
{
	if (r){
		if (r->standings) cudaFree(r->standings);
		if (r->vsChamp) cudaFree(r->vsChamp);
		free(r);
	}
}

void dumpResultsAux(FILE *f, WON_LOSS *standings, WON_LOSS *vsChamp)
{
//	printf("dumpResultsAux for standings at %p and vsChamp at %p ... \n", standings, vsChamp);
	for (int iSession = g_p.standings_freq-1; iSession < g_p.num_sessions; iSession += g_p.standings_freq) {
		// sort the standings by agent number (vsChamp is already by agent number)
		qsort(standings + iSession * g_p.num_agents, g_p.num_agents, sizeof(WON_LOSS), wl_byagent);
		for (int iAg = 0; iAg < g_p.num_agents; iAg++) {
			unsigned iStand = iSession * g_p.num_agents + iAg;
			fprintf(f, "%d, %d, %d, %d, %d, %d, %d, %d, %f, %f, %d\n", iSession, iAg, standings[iStand].games, standings[iStand].wins, standings[iStand].losses, vsChamp[iStand].games, vsChamp[iStand].wins, vsChamp[iStand].losses, standings[iStand].alpha, standings[iStand].lambda, standings[iStand].training_pieces);
		}
	}
//	printf("done.\n");
}

void dumpResults(RESULTS *r)
{
//	printf("dumpResults ... ");
	FILE *f = fopen(LEARNING_LOG_FILE, "w");
	if (!f) {
		printf("could not open the LEARNING_LOG_FILE %s\n", LEARNING_LOG_FILE);
		return;
	}
	
	save_parameters(f);
	
	//	// loop through each session and save the won-loss data to the LEARNING_LOG_FILE
	//	for (int iSession = 0; iSession < g_p.num_sessions; iSession++) {
	//		// sort the standings by agent number (vsChamp is already by agent number)
	//		qsort(r->standings + iSession * g_p.num_agents, g_p.num_agents, sizeof(WON_LOSS), wl_byagent);
	//		for (int iAg = 0; iAg < g_p.num_agents; iAg++) {
	//			unsigned iStand = iSession * g_p.num_agents + iAg;
	//			fprintf(f, "%d, %d, %d, %d, %d, %d, %d, %d\n", iSession, iAg, r->standings[iStand].games, r->standings[iStand].wins, r->standings[iStand].losses, r->vsChamp[iStand].games, r->vsChamp[iStand].wins, r->vsChamp[iStand].losses);
	//		}
	//	}
	dumpResultsAux(f, r->standings, r->vsChamp);
	fclose(f);
	
//	printf("done.\n");
}

WON_LOSS *copy_standings_to_CPU(RESULTS *rGPU)
{
	WON_LOSS *standings = (WON_LOSS *)malloc(g_p.num_sessions * g_p.num_agents * sizeof(WON_LOSS));
	CUDA_SAFE_CALL(cudaMemcpy(standings, rGPU->standings, g_p.num_sessions * g_p.num_agents * sizeof(WON_LOSS), cudaMemcpyDeviceToHost));
	return standings;
}

WON_LOSS *copy_vsChamp_to_CPU(RESULTS *rGPU)
{
	WON_LOSS *vsChamp = (WON_LOSS *)malloc(g_p.num_sessions * g_p.num_agents * sizeof(WON_LOSS));
	CUDA_SAFE_CALL(cudaMemcpy(vsChamp, rGPU->vsChamp, g_p.num_sessions * g_p.num_agents * sizeof(WON_LOSS), cudaMemcpyDeviceToHost));
	return vsChamp;
}

unsigned dumpResultsGPU(RESULTS *rGPU)
{
	// copy the standings and vsChamps values to host memory
//	WON_LOSS *standings = (WON_LOSS *)malloc(g_p.num_sessions * g_p.num_agents * sizeof(WON_LOSS));
//	WON_LOSS *vsChamp = (WON_LOSS *)malloc(g_p.num_sessions * g_p.num_agents * sizeof(WON_LOSS));
//	CUDA_SAFE_CALL(cudaMemcpy(standings, rGPU->standings, g_p.num_sessions * g_p.num_agents * sizeof(WON_LOSS), cudaMemcpyDeviceToHost));
//	CUDA_SAFE_CALL(cudaMemcpy(vsChamp, rGPU->vsChamp, g_p.num_sessions * g_p.num_agents * sizeof(WON_LOSS), cudaMemcpyDeviceToHost));
	
	// file name will have random number at the end to prevent over-writing old files
	char filename_buffer[4096];
	srandom(clock());
	unsigned ii = random();
	snprintf(filename_buffer, 4096, "%s%d.csv", LEARNING_LOG_FILE_GPU, ii);

	WON_LOSS *standings = copy_standings_to_CPU(rGPU);
	WON_LOSS *vsChamp = copy_vsChamp_to_CPU(rGPU);
	
	FILE *f = fopen(filename_buffer, "w");
	if (!f) {
		printf("could not open the LEARNING_LOG_FILE_GPU %s\n", LEARNING_LOG_FILE_GPU);
		return 0;
	}
	
	save_parameters(f);
	dumpResultsAux(f, standings, vsChamp);
	fclose(f);
	
	free(standings);
	free(vsChamp);
	return ii;
}

// Print sorted standings using the WON_LOSS information in standings (from learning vs. peers),
// and vsChamp (from competing against benchmark agent).
void print_standings_aux(unsigned *training_pieces, float *alpha, float *lambda, WON_LOSS *standings, WON_LOSS *vsChamp)
{
	unsigned printBenchmark = 0 < vsChamp[0].games;
	qsort(standings, g_p.num_agents, sizeof(WON_LOSS), wl_compare);
	printf(    "                               G     W     L    PCT");
	if (printBenchmark) printf("    %4d games vs Champ\n", g_p.benchmark_games);
	else printf("\n");
	
	WON_LOSS totChamp = {0, 0, 0, 0};
	WON_LOSS totStand = {0, 0, 0, 0};
	
	for (int i = 0; i < g_p.num_agents; i++) {
		//			printf("agent%4d  %4d %4d %4d  %5.3f", standings[i].agent, standings[i].games, standings[i].wins, standings[i].losses, 0.5f * (1.0f + (float)(standings[i].wins - standings[i].losses) / (float)standings[i].games));
		unsigned iAgent = standings[i].agent;

//		printf("standings place %d is agent %d\n", i,iAgent);
//		printf("   training_pieces %d\n", ag->training_pieces[iAgent]);
//		printf("             games %d\n", standings[i].games);
//		printf("               won %d\n", standings[i].wins);
//		printf("              loss %d\n", standings[i].losses);

		printf("agent%4d[p%2d a%4.2f l%4.2f] %6u %6u %6u  %5.3f", iAgent, training_pieces[iAgent], alpha[iAgent], lambda[iAgent], standings[i].games, standings[i].wins, standings[i].losses, winpct(standings[i]));
		
		totStand.games += standings[i].games;
		totStand.wins += standings[i].wins;
		totStand.losses += standings[i].losses;
		
		if (printBenchmark) {
			printf("  (%5u-%5u)  %+6d\n", vsChamp[iAgent].wins,vsChamp[iAgent].losses, (int)vsChamp[iAgent].wins - (int)vsChamp[iAgent].losses);
			totChamp.games += vsChamp[iAgent].games;
			totChamp.wins += vsChamp[iAgent].wins;
			totChamp.losses += vsChamp[iAgent].losses;
		}else printf("\n");
	}
	printf("                  avg     %7u%7u%7u  %5.3f ", totStand.games, totStand.wins, totStand.losses, winpct(totStand));
	if (printBenchmark) printf("(%6.1f-%6.1f)   %+6.1f\n", (float)totChamp.wins / (float)g_p.num_agents, (float)totChamp.losses / (float)g_p.num_agents, (float)((int)totChamp.wins-(int)totChamp.losses) / (float)g_p.num_agents);
	else printf("\n");
}

void print_standingsGPU(AGENT *agGPU, WON_LOSS *standings, WON_LOSS *vsChamp)
{
	unsigned *training_pieces = host_copyui(agGPU->training_pieces, g_p.num_agents);
	float *alpha = host_copyf(agGPU->alpha, g_p.num_agents);
	float *lambda = host_copyf(agGPU->lambda, g_p.num_agents);
	print_standings_aux(training_pieces, alpha, lambda, standings, vsChamp);
	free(training_pieces);
	free(alpha);
	free(lambda);
}

void print_standingsCPU(AGENT *agCPU, WON_LOSS *standings, WON_LOSS *vsChamp)
{
	print_standings_aux(agCPU->training_pieces, agCPU->alpha, agCPU->lambda, standings, vsChamp);
}
//void print_standingsGPU(WON_LOSS *standings, WON_LOSS *vsChamp)
//{
//	// copy the device standings to the host, then call the normal print standings function
//	WON_LOSS *h_standings = (WON_LOSS *)malloc(sizeof(WON_LOSS));
//	WON_LOSS *h_vsChamp = (WON_LOSS *)malloc(sizeof(WON_LOSS));
//	CUDA_SAFE_CALL(cudaMemcpy(h_standings, standings, g_p.num_agents * sizeof(WON_LOSS), cudaMemcpyDeviceToHost));
//	CUDA_SAFE_CALL(cudaMemcpy(h_vsChamp, vsChamp, g_p.num_agents * sizeof(WON_LOSS), cudaMemcpyDeviceToHost));
//	
//	print_standings(h_standings, h_vsChamp);
//	free(h_standings);
//	free(h_vsChamp);
//}
//

