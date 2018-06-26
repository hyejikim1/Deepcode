r1 = X_test_noise[:,:,0] # b_i
n1 = X_test_noise[:,:,1] # N_i
n2 = X_test_noise[:,:,2] # M_{i-1}
n3 = X_test_noise[:,:,3] # O_{i-1}

ttt = 20 # Number of sample points
rr1 = r1[0:ttt,:] # b_i
nn1 = n1[0:ttt,:] # N_i
nn2 = n2[0:ttt,:] # M_{i-1}
nn3 = n3[0:ttt,:] # O_{i-1}

p1 = codewords[:,:,1] 
p2 = codewords[:,:,2] 

pp1 = p1[0:ttt,:] # Parity1_i
pp2 = p2[0:ttt,:] # Parity2_i



plt.close()
plt.plot(nn1[rr1==0],pp1[rr1==0],'r.')
plt.plot(nn1[rr1==1],pp1[rr1==1],'bx')
plt.savefig('figs/SNR'+str(nsSNR)+'plot'+str(ttt)+'_PhaseI_noise_vs_parity1.png')


plt.close()
plt.plot(nn1[rr1==0],pp2[rr1==0],'r.')
plt.plot(nn1[rr1==1],pp2[rr1==1],'bx')
plt.savefig('figs/SNR'+str(nsSNR)+'plot'+str(ttt)+'_PhaseI_noise_vs_parity2.png')


plt.close()
plt.plot(pp1[rr1==0],pp2[rr1==0],'r.')
plt.plot(pp1[rr1==1],pp2[rr1==1],'bx')
plt.savefig('figs/SNR'+str(nsSNR)+'plot_'+str(ttt)+'_parity1_vs_parity2.png')
