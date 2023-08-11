begin

;;ERA5================================================	
;;降水量===============================================
	f1p			= addfile("/home/yangsong3/Extension/ERA5/ERA5-monthly/surface/prect_mon_1x1_1950_202106.nc","r")
	tp 			= dble2flt(f1p->tp)
	timeARRp 	= cd_calendar(f1p->time,-5) ;读取时间的位置
	yrARRp 		= timeARRp(:,0)
	monARRp 		= timeARRp(:,1)
	timeINDp1 	= ind(yrARRp.ge.1980.and.yrARRp.le.2020 .and. monARRp.eq.1)
	; timeINDp10 	= ind(yrARRp.ge.1979.and.yrARRp.le.2019 .and. monARRp.eq.10)
	timeINDp11 	= ind(yrARRp.ge.1979.and.yrARRp.le.2019 .and. monARRp.eq.11)
	timeINDp12 	= ind(yrARRp.ge.1979.and.yrARRp.le.2019 .and. monARRp.eq.12)

	preM  		= new((/3,181,360/),"float")
	; preM(0,:,:)	= dim_avg_n_Wrap(tp(timeINDp10,:,:),0)
	preM(0,:,:)	= dim_avg_n_Wrap(tp(timeINDp11,:,:),0)
	preM(1,:,:)	= dim_avg_n_Wrap(tp(timeINDp12,:,:),0)
	preM(2,:,:)	= dim_avg_n_Wrap(tp(timeINDp1,:,:),0)
	preM 		= preM*1000
	printVarSummary(preM)
	
	pre2021 	= tp(850:852,:,:) ;202010-202101
	pre2021 	= pre2021*1000
	prebias	= pre2021
	prebias	= pre2021 - preM
	predraw 	= dim_avg_n_Wrap(prebias,0) ;4个月平均
	printVarSummary(predraw)
	printMinMax(predraw({0:40},{95:150}),True) ;min=-3.76737   max=9.51831

;;蒸发量===============================================	
	flh			= addfile("/home/yangsong3/Extension/ERA5/ERA5-monthly/surface/surface_latent_heat_flux_mon_1x1_1950_202106.nc","r")
	lh 			= dble2flt(flh->slhf)
	timeARRe 	= cd_calendar(flh->time,-5) ;读取时间的位置
	yrARRe 		= timeARRe(:,0)
	monARRe 		= timeARRe(:,1)
	timeINDe1 	= ind(yrARRe.ge.1980.and.yrARRe.le.2020 .and. monARRe.eq.1)
	; timeINDe10	= ind(yrARRe.ge.1979.and.yrARRe.le.2019 .and. monARRe.eq.10)
	timeINDe11 	= ind(yrARRe.ge.1979.and.yrARRe.le.2019 .and. monARRe.eq.11)
	timeINDe12 	= ind(yrARRe.ge.1979.and.yrARRe.le.2019 .and. monARRe.eq.12)
	
	EM 			= new((/3,181,360/),"float")
	; EM(0,:,:)	= dim_avg_n_Wrap(lh(timeINDe10,:,:),0)
	EM(0,:,:)	= dim_avg_n_Wrap(lh(timeINDe11,:,:),0)
	EM(1,:,:)	= dim_avg_n_Wrap(lh(timeINDe12,:,:),0)
	EM(2,:,:)	= dim_avg_n_Wrap(lh(timeINDe1,:,:),0)
	EM 			= -1.0*EM/2510400 ;单位转换：J/m2 --> mm/day
	printVarSummary(EM)

	E2021 		= lh(850:852,:,:)
	E2021 		= -1.0*E2021/2510400 ;单位转换：J/m2 --> mm/day
	Ebias 		= E2021
	Ebias 		= E2021 - EM
	Edraw 		= dim_avg_n_Wrap(Ebias,0)
	printVarSummary(Edraw)
	printMinMax(Edraw({0:40},{95:150}),True) ;min=-1.8574   max=4.04335

;;=====================================================
	fu			= addfile("/home/yangsong3/Extension/ERA5/ERA5-monthly/pressure/uwind_mon_1x1_1950_202106.nc","r")
	fv			= addfile("/home/yangsong3/Extension/ERA5/ERA5-monthly/pressure/vwind_mon_1x1_1950_202106.nc","r")
	fw     	= addfile("/home/yangsong3/Extension/ERA5/ERA5-monthly/pressure/omega_mon_1x1_1950_202106.nc","r")
	fq			= addfile("/home/yangsong3/Extension/ERA5/ERA5-monthly/pressure/q_mon_1x1_1950_202106.nc","r")
	fsp		= addfile("/home/yangsong3/Extension/ERA5/ERA5-monthly/surface/sur_pressure_mon_1x1_1950_202106.nc","r")
	u			= dble2flt(fu->u(:,{1000:100},:,:))
	v			= dble2flt(fv->v(:,{1000:100},:,:))
	w			= dble2flt(fw->w(:,{1000:100},:,:))
	q			= dble2flt(fq->q(:,{1000:100},:,:))
	sp			= dble2flt(fsp->sp)
	sp 		= sp*0.01 

	;; u风速 (lev = 12)===================================
	timeARRu 	= cd_calendar(fu->time,-5) ;读取时间的位置
	yrARRu 		= timeARRu(:,0)
	monARRu 		= timeARRu(:,1)
	timeINDu1 	= ind(yrARRu.ge.1980.and.yrARRu.le.2020 .and. monARRu.eq.1)
	timeINDu10 	= ind(yrARRu.ge.1979.and.yrARRu.le.2019 .and. monARRu.eq.10)
	timeINDu11 	= ind(yrARRu.ge.1979.and.yrARRu.le.2019 .and. monARRu.eq.11)
	timeINDu12 	= ind(yrARRu.ge.1979.and.yrARRu.le.2019 .and. monARRu.eq.12)
	
	uM  			= new((/3,12,181,360/),"float")
	; uM(0,:,:,:)	= dim_avg_n_Wrap(u(timeINDu10,:,:,:),0)
	uM(0,:,:,:)	= dim_avg_n_Wrap(u(timeINDu11,:,:,:),0)
	uM(1,:,:,:)	= dim_avg_n_Wrap(u(timeINDu12,:,:,:),0)
	uM(2,:,:,:)	= dim_avg_n_Wrap(u(timeINDu1,:,:,:),0)
	uMM 			= dim_avg_n_Wrap(uM,0) ;[lev]x[lat]x[lon]
	printVarSummary(uMM)
	
	u2021		= u(850:852,:,:,:)
	du 		= u2021
	du 		= u2021 - uM
	duM 		= dim_avg_n_Wrap(du,0)
	printVarSummary(duM)

	;; v风速 (lev = 12)====================================
	timeARRv 	= cd_calendar(fv->time,-5) ;读取时间的位置
	yrARRv 		= timeARRv(:,0)
	monARRv 		= timeARRv(:,1)
	timeINDv1 	= ind(yrARRv.ge.1980.and.yrARRv.le.2020 .and. monARRv.eq.1)
	timeINDv10 	= ind(yrARRv.ge.1979.and.yrARRv.le.2019 .and. monARRv.eq.10)
	timeINDv11 	= ind(yrARRv.ge.1979.and.yrARRv.le.2019 .and. monARRv.eq.11)
	timeINDv12 	= ind(yrARRv.ge.1979.and.yrARRv.le.2019 .and. monARRv.eq.12)
	
	vM  			= new((/3,12,181,360/),"float")
	; vM(0,:,:,:)	= dim_avg_n_Wrap(v(timeINDv10,:,:,:),0)
	vM(0,:,:,:)	= dim_avg_n_Wrap(v(timeINDv11,:,:,:),0)
	vM(1,:,:,:)	= dim_avg_n_Wrap(v(timeINDv12,:,:,:),0)
	vM(2,:,:,:)	= dim_avg_n_Wrap(v(timeINDv1,:,:,:),0)
	vMM 			= dim_avg_n_Wrap(vM,0) ;[lev]x[lat]x[lon]
	printVarSummary(vMM)

	v2021		= v(850:852,:,:,:)
	dv 		= v2021
	dv 		= v2021 - vM
	dvM 		= dim_avg_n_Wrap(dv,0)
	printVarSummary(dvM)

	;; w垂直速度 (lev = 12)=====================================
	timeARRw 	= cd_calendar(fw->time,-5) ;读取时间的位置
	yrARRw 		= timeARRw(:,0)
	monARRw 		= timeARRw(:,1)
	timeINDw1 	= ind(yrARRw.ge.1980.and.yrARRw.le.2020 .and. monARRw.eq.1)
	timeINDw10 	= ind(yrARRw.ge.1979.and.yrARRw.le.2019 .and. monARRw.eq.10)
	timeINDw11 	= ind(yrARRw.ge.1979.and.yrARRw.le.2019 .and. monARRw.eq.11)
	timeINDw12 	= ind(yrARRw.ge.1979.and.yrARRw.le.2019 .and. monARRw.eq.12)
	
	wM  			= new((/3,12,181,360/),"float")
	; wM(0,:,:,:)	= dim_avg_n_Wrap(w(timeINDw10,:,:,:),0)
	wM(0,:,:,:)	= dim_avg_n_Wrap(w(timeINDw11,:,:,:),0)
	wM(1,:,:,:)	= dim_avg_n_Wrap(w(timeINDw12,:,:,:),0)
	wM(2,:,:,:)	= dim_avg_n_Wrap(w(timeINDw1,:,:,:),0)
	wMM 			= dim_avg_n_Wrap(wM,0) ;[lev]x[lat]x[lon]
	printVarSummary(wMM)

	w2021		= w(850:852,:,:,:)
	dw 		= w2021
	dw 		= w2021 - wM
	dwM 		= dim_avg_n_Wrap(dw,0)
	printVarSummary(dwM)

	;; 比湿q (lev = 12)=====================================
	timeARRq 	= cd_calendar(fq->time,-5) ;读取时间的位置
	yrARRq  		= timeARRq(:,0)
	monARRq 		= timeARRq(:,1)
	timeINDq1 	= ind(yrARRq.ge.1980.and.yrARRq.le.2020 .and. monARRq.eq.1)
	timeINDq10 	= ind(yrARRq.ge.1979.and.yrARRq.le.2019 .and. monARRq.eq.10)
	timeINDq11 	= ind(yrARRq.ge.1979.and.yrARRq.le.2019 .and. monARRq.eq.11)
	timeINDq12 	= ind(yrARRq.ge.1979.and.yrARRq.le.2019 .and. monARRq.eq.12)

	qM  			= new((/3,12,181,360/),"float")
	; qM(0,:,:,:)	= dim_avg_n_Wrap(q(timeINDq10,:,:,:),0)
	qM(0,:,:,:)	= dim_avg_n_Wrap(q(timeINDq11,:,:,:),0)
	qM(1,:,:,:)	= dim_avg_n_Wrap(q(timeINDq12,:,:,:),0)
	qM(2,:,:,:)	= dim_avg_n_Wrap(q(timeINDq1,:,:,:),0)
	qMM 			= dim_avg_n_Wrap(qM,0) ;[lev]x[lat]x[lon]
	printVarSummary(qMM)

	q2021		= q(850:852,:,:,:)
	dq 		= q2021
	dq 		= q2021 - qM
	dqM 		= dim_avg_n_Wrap(dq,0)
	printVarSummary(dqM)

	;; 地面气压sp===========================================
	timeARRsp 	= cd_calendar(fsp->time,-5) ;读取时间的位置
	yrARRsp 		= timeARRsp(:,0)
	monARRsp		= timeARRsp(:,1)
	timeINDsp1 	= ind(yrARRsp.ge.1980.and.yrARRsp.le.2020 .and. monARRsp.eq.1)
	timeINDsp10 = ind(yrARRsp.ge.1979.and.yrARRsp.le.2019 .and. monARRsp.eq.10)
	timeINDsp11 = ind(yrARRsp.ge.1979.and.yrARRsp.le.2019 .and. monARRsp.eq.11)
	timeINDsp12 = ind(yrARRsp.ge.1979.and.yrARRsp.le.2019 .and. monARRsp.eq.12)

	spM  			= new((/3,181,360/),"float")
	; spM(0,:,:)	= dim_avg_n_Wrap(sp(timeINDsp10,:,:),0)
	spM(0,:,:)	= dim_avg_n_Wrap(sp(timeINDsp11,:,:),0)
	spM(1,:,:)	= dim_avg_n_Wrap(sp(timeINDsp12,:,:),0)
	spM(2,:,:)	= dim_avg_n_Wrap(sp(timeINDsp1,:,:),0)
	spMM 			= dim_avg_n_Wrap(spM,0) ;[lev]x[lat]x[lon]
	printVarSummary(spMM)

;;垂直水汽===============================================
	p 		= u&level	;[12]
	dqdp 	= center_finite_diff_n(dq,p,False,0,1) ;[month]x[lev]x[lat]x[lon]
	copy_VarCoords(dq,dqdp)
	printVarSummary(dqdp)
	print(dqdp(0,:,0,0))

	Cdqdp	= center_finite_diff_n(qMM,p,False,0,0) ;[lev]x[lat]x[lon]
	copy_VarCoords(qMM,Cdqdp)
	printVarSummary(Cdqdp)
	print(Cdqdp(:,0,0))

;;水平水汽===============================================
	dqdx 	= new(dimsizes(dq),"float")  ;[month]x[lev]x[lat]x[lon]
	dqdy 	= new(dimsizes(dq),"float")
	Cdqdx 	= new(dimsizes(qMM),"float") ;[lev]x[lat]x[lon]
	Cdqdy 	= new(dimsizes(qMM),"float")
	
	LAT 	= dq&latitude
	LON 	= dq&longitude
	dlon 	= (LON(2)-LON(1))*0.0174533
	dlat 	= (LAT(2)-LAT(1))*0.0174533
	do i = 0,dimsizes(LAT)-1
		dx 	= 6378388.*cos(0.0174533*LAT(i))*dlon
		dqdx(:,:,i,:) 	= center_finite_diff_n(dq(:,:,i,:),dx,False,0,2)
		Cdqdx(:,i,:) 	= center_finite_diff_n(qMM(:,i,:),dx,False,0,1)
	end do  
	copy_VarCoords(dq,dqdx)
	copy_VarCoords(qMM,Cdqdx)
	do i = 0,dimsizes(LON)-1
		dy 	= 6378388.*dlat
		dqdy(:,:,:,i) 	= center_finite_diff_n(dq(:,:,:,i),dy,False,0,2)
		Cdqdy(:,:,i) 	= center_finite_diff_n(qMM(:,:,i),dy,False,0,1)
	end do
	copy_VarCoords(dq,dqdy)
	copy_VarCoords(qMM,Cdqdx)
	printVarSummary(dqdx)
	print(dqdx(0,:,0,0))
	print(dqdy(0,:,0,0))

;;计算水汽收支的偏差方程各项===========================
	Te1=uM   ;;[month]x[lev|12]x[lat]x[lon]
	Te2=uM
	Te3=qM	 ;;[month]x[lev|12]x[lat]x[lon]
	Te4=qM
	do i = 0,2
		Te1(i,:,:,:)=-1.0*wMM*dqdp(i,:,:,:)  ;垂直水汽输送热力项
		Te2(i,:,:,:)=-1.0*dw(i,:,:,:)*Cdqdp(:,:,:)  ;垂直水汽输送动力项
		Te3(i,:,:,:)=-1.0*uMM*dqdx(i,:,:,:)-1.0*vMM*dqdy(i,:,:,:)  ;水平水汽输送热力项
		Te4(i,:,:,:)=-1.0*du(i,:,:,:)*Cdqdx-1.0*dv(i,:,:,:)*Cdqdy  ;水平水汽输送动力项
	end do
	copy_VarCoords(dw,Te1)
	copy_VarCoords(dw,Te2)
	copy_VarCoords(dq,Te3)
	copy_VarCoords(dq,Te4)
	printVarSummary(Te1)	
	printVarSummary(Te3)
	print(Te1(0,:,0,0))
	print(Te2(0,:,0,0))
	; print(Te3(0,:,0,0))
	print(Te4(0,:,0,0))

;;垂直积分===============================================
	ptop 	= min(p)
	dp 		= dpres_plevel(p, spM, ptop, 0) ;[month]x[lev]x[lat]x[lon]
	printVarSummary(dp)
	print(dp(1,:,0,0))
	
	Tee1=Te1*dp
	Tee2=Te2*dp
	Tee3=Te3*dp
	Tee4=Te4*dp
	T1=dim_sum_n_Wrap(Tee1,1) ;[month]x[lat]x[lon]
	T2=dim_sum_n_Wrap(Tee2,1)
	T3=dim_sum_n_Wrap(Tee3,1)
	T4=dim_sum_n_Wrap(Tee4,1)
	T1=T1/9.81*86400  ;;单位换算成mm/day
	T2=T2/9.81*86400
	T3=T3/9.81*86400*100
	T4=T4/9.81*86400*100
	copy_VarCoords(dw(:,0,:,:),T1)
	copy_VarCoords(dw(:,0,:,:),T2)
	copy_VarCoords(dq(:,0,:,:),T3)
	copy_VarCoords(dq(:,0,:,:),T4)
	print(T1(:,0,0))  
	print(T2(:,0,0))
	print(T3(:,0,0))
	print(T4(:,0,0))

	T1draw = dim_avg_n_Wrap(T1,0) ;[lat]x[lon]
	T2draw = dim_avg_n_Wrap(T2,0)
	T3draw = dim_avg_n_Wrap(T3,0)
	T4draw = dim_avg_n_Wrap(T4,0)
	printMinMax(T1draw({0:40},{95:150}),True) ;min=-2.85205   max=4.00643	
	printMinMax(T2draw({0:40},{95:150}),True) ;min=-11.0172   max=18.0444
	printMinMax(T3draw({0:40},{95:150}),True) ;min=-6.84386   max=5.86864
	printMinMax(T4draw({0:40},{95:150}),True) ;min=-4.3223   max=4.08617

	output="/home/ys17-23/Extension/personal-data/chenhj/SAM_EAM_data/new_paper_data/qbudgetanalysis/"
   system("rm -f "+output+"ERA5_ncl_qbudget_test_1x1.nc")
   fout = addfile(output+"ERA5_ncl_qbudget_test_1x1.nc","c")
   fout->pre  = predraw
   fout->E    = Edraw
   fout->T1   = T1draw
   fout->T2   = T2draw
   fout->T3   = T3draw
   fout->T4   = T4draw

; ;;画图，展板=============================================
; wks = gsn_open_wks("pdf","Fig2-202010-202101-ERA5")
; plot = new(6,graphic)

; res                      = True
; res@gsnAddCyclic		 = False
; res@cnFillOn             = True
; res@cnFillPalette       = "precip_diff_12lev"
; res@cnLevelSelectionMode = "ExplicitLevels"   ; set explicit contour levels
; res@cnLevels             = (/-5,-3,-1,-0.7,-0.4,-0.1,0.1,0.4,0.7,1,3,5/)
; res@cnFillColors =(/0,1,2,3,4,5,6,7,8,9,10,11,12/)
; ; res@cnLevelSelectionMode = "ManualLevels"  ; set explicit contour levels   
; ; res@cnMinLevelValF = -6
; ; res@cnMaxLevelValF = 6  
; ; res@cnLevelSpacingF = 1  ;10 
; res@gsnDraw              = False
; res@gsnFrame             = False
; res@mpDataBaseVersion     = "LowRes"        ; better
; res@mpDataSetName         = "$NCARG_ROOT/lib/ncarg/database1/Earth..4"
; res@mpOutlineBoundarySets = "Geophysical"
; res@pmTickMarkDisplayMode = "Always"           
; res@mpOutlineOn           = True			; 显示国界线
; res@cnInfoLabelOn = False
; res@cnLineLabelsOn  = False
; res@cnLinesOn       = False
; ; res@mpGeophysicalLineColor      = "gray35"
; res@mpGeophysicalLineThicknessF      = 1.5
; res@mpOceanFillColor      = -1
; res@txFontHeightF = 0.02
; res@tmXBLabelFontHeightF = 0.02
; res@tmYLLabelFontHeightF = 0.02
; res@lbLabelBarOn =False
; res@gsnLeftStringFontHeightF = 0.026
; res@gsnRightStringFontHeightF = 0.026
; res@mpMinLatF    = 0.
; res@mpMaxLatF    = 40.
; res@mpMinLonF    = 95.
; res@mpMaxLonF    = 150.

; res@gsnLeftString  = "(a) P'"
; res@gsnRightString  = "mm day~S~-1~N~"
; plot(0) = gsn_csm_contour_map_ce(wks,predraw,res) ;蒸发项

; res@gsnLeftString  = "(b) E'"
; res@gsnRightString  = "mm day~S~-1~N~"
; plot(1) = gsn_csm_contour_map_ce(wks,Edraw,res) ;蒸发项

; res@gsnLeftString  = "(c) -<wdq'>"
; res@gsnRightString  = "mm day~S~-1~N~"
; plot(2) = gsn_csm_contour_map_ce(wks,T1draw,res) ;垂直水汽输送热力项

; res@gsnLeftString  = "(d) -<w'dq>"
; res@gsnRightString  = "mm day~S~-1~N~"
; plot(3) = gsn_csm_contour_map_ce(wks,T2draw,res) ;垂直水汽输送动力项

; res@gsnLeftString  = "(e) -<vdq'>"
; res@gsnRightString  = "mm day~S~-1~N~"
; plot(4) = gsn_csm_contour_map_ce(wks,T3draw,res) ;水平水汽输送热力项

; res@gsnLeftString  = "(f) -<v'dq>"
; res@gsnRightString  = "mm day~S~-1~N~"
; plot(5) = gsn_csm_contour_map_ce(wks,T4draw,res) ;水平水汽输送动力项


; ; res@gsnLeftString  = "(a) E'"
; ; res@gsnRightString  = "mm day~S~-1~N~"
; ; plot(0) = gsn_csm_contour_map_ce(wks,Edraw,res) ;蒸发项

; ; res@gsnLeftString  = "(b) -<wdq'>"
; ; res@gsnRightString  = "mm day~S~-1~N~"
; ; plot(1) = gsn_csm_contour_map_ce(wks,T1draw,res) ;垂直水汽输送热力项

; ; res@gsnLeftString  = "(c) -<w'dq>"
; ; res@gsnRightString  = "mm day~S~-1~N~"
; ; plot(2) = gsn_csm_contour_map_ce(wks,T2draw,res) ;垂直水汽输送动力项

; ; res@gsnLeftString  = "(d) -<vdq'>"
; ; res@gsnRightString  = "mm day~S~-1~N~"
; ; plot(3) = gsn_csm_contour_map_ce(wks,T3draw,res) ;水平水汽输送热力项

; ; res@gsnLeftString  = "(e) -<v'dq>"
; ; res@gsnRightString  = "mm day~S~-1~N~"
; ; plot(4) = gsn_csm_contour_map_ce(wks,T4draw,res) ;水平水汽输送动力项

; ypt = (/22, 26, 31, 31, 22, 22/)
; xpt = (/108, 108, 116, 122, 122, 108/)
; resp                  = True                      ; polyline mods desired
; resp@gsLineColor      = "red"                     ; color of lines
; resp@gsLineThicknessF = 5.0                       ; thickness of lines
; resp@gsLineLabelString = ""                    ; adds a line label string
; resp@gsLineDashPattern = 1
; dum1 = new(5,graphic)
; dum2 = new(5,graphic)
; dum3 = new(5,graphic)
; dum4 = new(5,graphic)
; dum5 = new(5,graphic)
; dum6 = new(5,graphic)
; do i = 0 , 4
;    dum1(i)=gsn_add_polyline(wks,plot(0),xpt(i:i+1),ypt(i:i+1),resp)
;    dum2(i)=gsn_add_polyline(wks,plot(1),xpt(i:i+1),ypt(i:i+1),resp)
;    dum3(i)=gsn_add_polyline(wks,plot(2),xpt(i:i+1),ypt(i:i+1),resp)
;    dum4(i)=gsn_add_polyline(wks,plot(3),xpt(i:i+1),ypt(i:i+1),resp)
;    dum5(i)=gsn_add_polyline(wks,plot(4),xpt(i:i+1),ypt(i:i+1),resp)
;    dum6(i)=gsn_add_polyline(wks,plot(5),xpt(i:i+1),ypt(i:i+1),resp)
; end do

; resP=True
; resP@gsnFrame      = False
; resP@gsnPanelBottom= 0.05
; resP@gsnPanelYWhiteSpacePercent = 5
; resP@gsnPanelFigureStringsPerimOn=False
; resP@gsnPanelLabelBar = True                ; add common colorbar
; resP@lbOrientation = "Vertical"
; resP@pmLabelBarWidthF     = 0.05
; resP@pmLabelBarHeightF    = 0.49
; resP@lbLabelFontHeightF   = 0.012
; gsn_panel(wks,plot,(/3,2/),resP)

; frame(wks)	
end