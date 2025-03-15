# revelio-marketing-turnover-brand-equity
The code for the project "Marketing employee turnover and brand performance: evidence from 477 firms"

Here is the work flow for running the codes.

directory = '/work/SafeGraph/revelio/code'

sdirectory = '/work/SafeGraph/revelio/data'

tdirectory = '/work/SafeGraph/revelio/data/yougov_new'

fdirectory = '/work/SafeGraph/revelio/data/2023latest'

fvdirectory = '/work/SafeGraph/revelio/data/yougov'


1. 0_yougov_brand_list.ipynb: this file uses the yougov data alone (ugov_cik.dta). It generates a public brand list for those with valid cik (453 firms) and their corresponding company name, and a private brand list
of brand names for those without valid cik (852 brands). Input: ugov_cik.dta. Output: public_brand_list.csv, private_brand_list.csv.

2. 0a_yougov_public-{0-2}.py: these files use the revelio job position files and the Revelio coompany reference files. This python file selects all job position histories that are in US and have a valid cik by matching through rcid. It generates a lists of public job history files with company information from company reference files. Input: company_ref_00{00-31}_part_00.parquet, Individual_Job_Positions_and_Job_History-{0-768}.csv.gz. Output: yougov_public_job_history-{0-48}.csv.

3. 0aa_yougov_public_list.py: this file uses public_brand_list.csv and yougov_public_job_history-{0-48}.csv to get a list of public brands that appear in both YouGov and Revelio. (401 firms) The "firm" column is the company name from the YouGov file. Input: 
public_brand_list.csv, yougov_public_job_history-{0-48}.csv. Output: public_brand_list_revelio_new.csv. 

4. 0b_yougov_private_direct.py: this file uses the private brand list generated above and the Revelio company reference files. It uses the brand name from yougov private brand list to match with company name or child company or ultimate parent company name (both original names and names without spaces) in the Revelio company reference file. This generates a list of private brands that can be matched directly. (307 firms) The "Brand" column is the brand column from the YouGov file. Input: private_brand_list.csv, company_ref_00{00-31}_part_00.parquet. Output: private_rcid_direct_2.csv, private_unmatched_brands_2.csv. 

5. 0c_yougov_private_clean_direct.py: this file uses the private brand direct list to select job histories from revelio job history files. Input: private_brand_rcid_direct_2.csv, individual_Job_Positions_and_Job_History-{0-768}.csv.gz. Output: yougov_private_direct_job_history-{0-3}.csv.

6. 0d_yougov_private_fuzzy-{0-10}.py: this file uses the private brand list generated above and the Revelio company reference files. For those private brands that do not have a direct matching, it uses fuzzy matching method  	to match the brand name from yougov private brand list and the company name or child company name or parent company name from Revelio company reference files. This generateas a list of private brands that can matched using 	highest fuzzy string matching ratio. Then I manually check and organize these lists. Most of them are obviously not good matches and some of them are similar names but refer to different brands. (112 firms) Input: private_brand_list.csv, private_rcid_direct_2.csv, private_unmatched_brands_2.csv, company_ref_00{00-31}_part_00.parquet.
Output: private_brand_rcid_fuzzy-{0-10}.csv. After manual pick, changed to private_brand_rcid_fuzzy_manual-{0-10}.csv.

7. 0d_yougov_private_fuzzy-11.py: this file uses large brands from the private brand list that do not have good fuzzy matches with the Revelio company reference files. I manually checked and picked 
the corresponding names from both sources. (26 brands, 21 firms) Input: company_ref_00{00-31}_part_00.parquet. Output: private_brand_rcid_fuzzy_manual-11.csv.

8. 0e_yougov_private_clean_fuzzy.py: this file uses the private brand fuzzy list to select job histories from revelio job history files. Input: private_brand_rcid_fuzzy_manual-{0-11}.csv, Individual_Job_Positions_and_Job_History-{0-768}.csv. Output: yougov_private_fuzzy_job_history-{0-3}.csv.

9. 0f_yougov_brand_employee_quarterly-{0-7}.py, 0f_yougov_brand_employee_quarterly_coombine.py: these files use all the public and private job history files to get the quarterly number of current employees, average tenure and average salary for each firm. Input: public_brand_list_revelio_new.csv, yougov_public_job_history-{0-48}.csv, yougov_private_direct_job_history-{0-3}.csv, yougov_private_fuzzy_job_history-{0-3}.csv. Output: brand_turnover_overall_quarterly.csv.

10. 0g_yougov_mkt_exe_job_history.py: this file uses all the public and private job history files generated above to select marketing employees, without specifying the seniority level. Input: public_brand_list_revelio_new.csv, yougov_public_job_history-{0-48}.csv, yougov_private_direct_job_history-{0-3}.csv, yougov_private_fuzzy_job_history-{0-3}.csv. Output: yougov_exe_job_history_no_seniority.csv.

11. 1_yougov_embedding.py: this file uses the selected job history files to assign text embeddings for all the unique job titles. Input: yougov_mkt_exe_job_history_no_seniority.csv. Output: unique_job_titles_with_embeddings.csv.

12. 1a_yougov_embedding_keyword_multi.py: this file uses the 2-step LLM method to assign digital roles and seniority levels to all the job histories. Input: yougov_mkt_exe_job_history_no_seniority.csv, unique_job_titles_with_embeddings.csv. Output: unique_job_titles_with_rule_based_seniority.csv, yougov_mkt_exe_job_history_digital.csv, complete_job_titles_digital.xlsx.

13. 2a_yougov_naics.py: this file gathers all the public and private brand lists, their naics codes, and the infomration on parent company. Then a dictionary of peers and peers-of-peers is constructed for all firms that appear in our brand list. Input: public_brand_list_revelio_new.csv, private_brand_rcid_direct_2.csv, private_brand_rcid_fuzzy_manual-{0-11}.csv, company_ref_00{00-31}_part_00.parquet. Output: peers_dict.json.

14. 2b_youg_peer_file.py: this file selects the job histories of all the marketing-related employees from all the firms that appear in the peers_dict.json for ease to load them. Input: peers_dict.json, Individual_Job_Positions_and_Job_History-{0-768}.csv.gz. Output: filtered_job_history_all_peers.csv.

15. 2c_peer_embedding-{0-9}.py: this file uses the dictionary file and the all peer job history file to get text embeddings for all unique job titles that appear in peers-of-peer firms. Input: filtered_job_history_all_peers.csv. Output: unique_job_titles_with_embeddings_peer-{0-9}.csv. 

16. 2d_peer_embedding_keyword_hybrid.py: this file uses the same 2-step LLM method to assign digital roles and seniority levels to all the job histories of peers-of-peer firms. Input: filtered_job_history_all_peers.csv, unique_job_titles_with_embeddings_peer-{0-9}.csv. Output: yougov_mkt_peer_job_history.csv, complete_peer_job_titles.xlsx.

17. 2e_yougov_peer.py: this file constructs the number of marketing employee hires and leaves at each seniority level for each firm at each day. Input: peers_dict.json, yougov_mkt_peer_job_history.csv, Output: aggregated_peer_hire.csv, aggregated_peer_peer_hire.csv, aggregated_peer_leave.csv, aggregated_peer_peer_leave.csv.

18. 3_yougov_hire_leave.py: this file uses the marketing executive job histories yougov_mkt_exe_job_history_digital.csv to construct the number of marketing employee hires and leaves at each seniority for each firm at each day. The hires and leaves are defined as moves between different firms. Input: yougov_mkt_exe_job_history_digital.csv. Output: aggregated_job_history_hire_new.csv, aggregated_job_history_leave_new.csv.

19. 4_yougov_analysis.ipynb: this file generates all the output files above to construct a panel data of brand-date with brand health metrics and firm turnover information, and peer-of-peer firm turnover information. Input: ugov_cik.dta, public_brand_list_revelio_new.csv, company_ref_00{str(0-31).zfill(2)}_part_00.parquet, private_brand_rcid_direct_2.csv, private_brand_rcid_fuzzy_manual-{0-11}.csv, aggregated_job_history_hire_new.csv, aggregated_job_history_leave_new.csv, aggregated_peer_hire.csv, aggregated_peer_peer_hire.csv, aggregated_peer_leave.csv, aggregated_peer_peer_leave.csv. Output: yougov_merged_data_new.csv. 

21. 5_yougov_balance_new.R: this file generates the firm-quarter balanced panel from the intitla highly unbalanced panel. Input: yougov_merged_data_new.csv. Output: 
