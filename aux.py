  print(no_nans.loc[:,['vert_max', 'vert_maxreach','vert_nostep', 'hand_length','hand_width']].isna().any())
  print(no_nans.loc[:,[
    'draft_pick', 'hght_noshoes','hght_wtshoes','wingspan','standing_reach',
    'weight', 'body_fat', 'clg_games_plyd', 'pts_ppg', 'rpg','ast', 'fg2_pct', 
    'fg3_pct', 'ft_pct', 'guards', 'forwards', 'centers', 'drafted', 'nba_gms_plyed']].isna().any())
  exit()

  corr_matrix=no_nans.corr().abs()
  sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        .stack()
        .sort_values(ascending=False))
  pd.set_option('display.max_rows', len(sol))
  print(sol)
  exit()
  count=1