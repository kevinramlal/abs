
'''
	Contains multi-purpose functions.
'''


def latex_table(df, caption="", label="", index=False):
	'''
		Prints DataFrame in format readable by LaTex.
	'''
	return "\\begin{table}[H]\n\centering\n"+df.to_latex(index=index)+"\caption{"+caption+"}\n\label{tab:"+label+"}\n\end{table}"

