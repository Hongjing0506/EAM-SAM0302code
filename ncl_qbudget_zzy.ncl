begin

;;ERA5================================================	
;;降水量===============================================
	f1p			= addfile("/home/yangsong3/Extension/ERA5/ERA5-monthly/surface/prect_mon_1x1_1950_202106.nc","r")
	tp 			= dble2flt(f1p->tp)
	timeARRp 	= cd_calendar(f1p->time,-5) ;读取时间的位置
	yrARRp 		= timeARRp(:,0)
	monARRp 		= timeARRp(:,1)
	timeINDp6 	= ind(yrARRp.ge.1979.and.yrARRp.le.2014 .and. monARRp.eq.6)
	timeINDp7 	= ind(yrARRp.ge.1979.and.yrARRp.le.2014 .and. monARRp.eq.7)
	timeINDp8 	= ind(yrARRp.ge.1979.and.yrARRp.le.2014 .and. monARRp.eq.8)

	preM  		= new((/3,181,360/),"float")
	preM(0,:,:)	= dim_avg_n_Wrap(tp(timeINDp6,:,:),0)  ;6月平均
	preM(1,:,:)	= dim_avg_n_Wrap(tp(timeINDp7,:,:),0)  ;7月平均 
	preM(2,:,:)	= dim_avg_n_Wrap(tp(timeINDp8,:,:),0)  ;8月平均
	preM 		= preM*1000   ;m/day为单位转换为mm/day为单位
	printVarSummary(preM)
	
  preJJA = new((/3,36,181,360/),"float")
	preJJA(0,:,:,:) 	= tp(timeINDp6,:,:) ;提取6月份数据
  preJJA(1,:,:,:) 	= tp(timeINDp7,:,:) ;提取7月份数据
  preJJA(2,:,:,:) 	= tp(timeINDp8,:,:) ;提取8月份数据

	preJJA 	= preJJA*1000
  prebias = new(dimsizes(preJJA), "float")
  copy_VarCoords(preJJA, prebias)

  ; 计算逐年的各月距平
  do i=0,35
    prebias(:,i,:,:) = preJJA(:,i,:,:)-preM
  end do
	predraw 	= dim_avg_n_Wrap(prebias,0) ;3个月平均
	printVarSummary(predraw)

;;蒸发量===============================================	
	flh			= addfile("/home/yangsong3/Extension/ERA5/ERA5-monthly/surface/surface_latent_heat_flux_mon_1x1_1950_202106.nc","r")
	lh 			= dble2flt(flh->slhf)
	timeARRe 	= cd_calendar(flh->time,-5) ;读取时间的位置
	yrARRe 		= timeARRe(:,0)
	monARRe 		= timeARRe(:,1)
	timeINDp6 	= ind(yrARRp.ge.1979.and.yrARRp.le.2014 .and. monARRp.eq.6)
	timeINDp7 	= ind(yrARRp.ge.1979.and.yrARRp.le.2014 .and. monARRp.eq.7)
	timeINDp8 	= ind(yrARRp.ge.1979.and.yrARRp.le.2014 .and. monARRp.eq.8)
	
	EM 			= new((/3,181,360/),"float")
	; EM(0,:,:)	= dim_avg_n_Wrap(lh(timeINDe10,:,:),0)
	EM(0,:,:)	= dim_avg_n_Wrap(lh(timeINDp6,:,:),0)
	EM(1,:,:)	= dim_avg_n_Wrap(lh(timeINDp7,:,:),0)
	EM(2,:,:)	= dim_avg_n_Wrap(lh(timeINDp8,:,:),0)
	EM 			= -1.0*EM/2510400 ;单位转换：J/m2 --> mm/day
	printVarSummary(EM)

  EJJA 			= new((/3,36,181,360/),"float")

	EJJA(0,:,:,:) 	= lh(timeINDp6,:,:) ;提取6月份数据
  EJJA(1,:,:,:) 	= lh(timeINDp7,:,:) ;提取7月份数据
  EJJA(2,:,:,:) 	= lh(timeINDp8,:,:) ;提取8月份数据

	EJJA 		= -1.0*EJJA/2510400 ;单位转换：J/m2 --> mm/day
	Ebias 		= new(dimsizes(EJJA), "float")
  copy_VarCoords(EJJA, Ebias)
  do i=0,35
    Ebias(:,i,:,:) = EJJA(:,i,:,:)-EM
  end do
	Edraw 		= dim_avg_n_Wrap(Ebias,0)  ;计算6-8月季节平均
	printVarSummary(Edraw)
	; printMinMax(Edraw({0:40},{95:150}),True) ;min=-1.8574   max=4.04335

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
	sp 		= sp*0.01 ;Pa为单位转换为hPa为单位

	;; u风速 (lev = 12)===================================
	timeARRu 	= cd_calendar(fu->time,-5) ;读取时间的位置
	yrARRu 		= timeARRu(:,0)
	monARRu 		= timeARRu(:,1)
	timeINDu6 	= ind(yrARRu.ge.1979.and.yrARRu.le.2014 .and. monARRu.eq.6)
	timeINDu7 	= ind(yrARRu.ge.1979.and.yrARRu.le.2014 .and. monARRu.eq.7)
	timeINDu8 	= ind(yrARRu.ge.1979.and.yrARRu.le.2014 .and. monARRu.eq.8)
	
	uM  			= new((/3,12,181,360/),"float") ;[month]x[lev]x[lat]x[lon]
	; uM(0,:,:,:)	= dim_avg_n_Wrap(u(timeINDu10,:,:,:),0)
	uM(0,:,:,:)	= dim_avg_n_Wrap(u(timeINDu6,:,:,:),0)
	uM(1,:,:,:)	= dim_avg_n_Wrap(u(timeINDu7,:,:,:),0)
	uM(2,:,:,:)	= dim_avg_n_Wrap(u(timeINDu8,:,:,:),0)
	uMM 			= dim_avg_n_Wrap(uM,0) ;[lev]x[lat]x[lon]
	printVarSummary(uMM)
	
  uJJA = new((/3,36,12,181,360/),"float")
  uJJA(0,:,:,:,:)	= u(timeINDu6,:,:,:)
	uJJA(1,:,:,:,:)	= u(timeINDu7,:,:,:)
	uJJA(2,:,:,:,:)	= u(timeINDu8,:,:,:)

	du 		= new(dimsizes(uJJA),"float")
  copy_VarCoords(uJJA, du)
  do i =0,35
    du(:,i,:,:,:) 		= uJJA(:,i,:,:,:) - uM
  end do
  
	duM 		= dim_avg_n_Wrap(du,0) ;计算月份平均，[year]x[lev]x[lat]x[lon]
	printVarSummary(duM)

	;; v风速 (lev = 12)====================================
	timeARRv 	= cd_calendar(fv->time,-5) ;读取时间的位置
	yrARRv 		= timeARRv(:,0)
	monARRv 		= timeARRv(:,1)
	timeINDv6 	= ind(yrARRv.ge.1979.and.yrARRv.le.2014 .and. monARRv.eq.6)
	timeINDv7 	= ind(yrARRv.ge.1979.and.yrARRv.le.2014 .and. monARRv.eq.7)
	timeINDv8 	= ind(yrARRv.ge.1979.and.yrARRv.le.2014 .and. monARRv.eq.8)
	
	vM  			= new((/3,12,181,360/),"float")
	; vM(0,:,:,:)	= dim_avg_n_Wrap(v(timeINDv10,:,:,:),0)
	vM(0,:,:,:)	= dim_avg_n_Wrap(v(timeINDv6,:,:,:),0)
	vM(1,:,:,:)	= dim_avg_n_Wrap(v(timeINDv7,:,:,:),0)
	vM(2,:,:,:)	= dim_avg_n_Wrap(v(timeINDv8,:,:,:),0)
	vMM 			= dim_avg_n_Wrap(vM,0) ;[lev]x[lat]x[lon]
	printVarSummary(vMM)

  vJJA = new((/3,36,12,181,360/),"float")
  vJJA(0,:,:,:,:)	= v(timeINDu6,:,:,:)
	vJJA(1,:,:,:,:)	= v(timeINDu7,:,:,:)
	vJJA(2,:,:,:,:)	= v(timeINDu8,:,:,:)


	dv 		= new(dimsizes(vJJA),"float")
  copy_VarCoords(vJJA, dv)
  do i =0,35
    dv(:,i,:,:,:) 		= vJJA(:,i,:,:,:) - vM
  end do
	dvM 		= dim_avg_n_Wrap(dv,0)
	printVarSummary(dvM)

	;; w垂直速度 (lev = 12)=====================================
	timeARRw 	= cd_calendar(fw->time,-5) ;读取时间的位置
	yrARRw 		= timeARRw(:,0)
	monARRw 		= timeARRw(:,1)
	timeINDw6 	= ind(yrARRw.ge.1979.and.yrARRw.le.2014 .and. monARRw.eq.6)
	timeINDw7 	= ind(yrARRw.ge.1979.and.yrARRw.le.2014 .and. monARRw.eq.7)
	timeINDw8 	= ind(yrARRw.ge.1979.and.yrARRw.le.2014 .and. monARRw.eq.8)
	
	wM  			= new((/3,12,181,360/),"float")
	; wM(0,:,:,:)	= dim_avg_n_Wrap(w(timeINDw10,:,:,:),0)
	wM(0,:,:,:)	= dim_avg_n_Wrap(w(timeINDw6,:,:,:),0)
	wM(1,:,:,:)	= dim_avg_n_Wrap(w(timeINDw7,:,:,:),0)
	wM(2,:,:,:)	= dim_avg_n_Wrap(w(timeINDw8,:,:,:),0)
	wMM 			= dim_avg_n_Wrap(wM,0) ;[lev]x[lat]x[lon]
	printVarSummary(wMM)

  wJJA = new((/3,36,12,181,360/),"float")
  wJJA(0,:,:,:,:)	= w(timeINDu6,:,:,:)
	wJJA(1,:,:,:,:)	= w(timeINDu7,:,:,:)
	wJJA(2,:,:,:,:)	= w(timeINDu8,:,:,:)


	dw 		= new(dimsizes(wJJA),"float")
  copy_VarCoords(wJJA, dw)
  do i =0,35
    dw(:,i,:,:,:) 		= wJJA(:,i,:,:,:) - wM
  end do
	dwM 		= dim_avg_n_Wrap(dw,0)
	printVarSummary(dwM)

	;; 比湿q (lev = 12)=====================================
	timeARRq 	= cd_calendar(fq->time,-5) ;读取时间的位置
	yrARRq  		= timeARRq(:,0)
	monARRq 		= timeARRq(:,1)
	timeINDq6 	= ind(yrARRq.ge.1979.and.yrARRq.le.2014 .and. monARRq.eq.6)
	timeINDq7 	= ind(yrARRq.ge.1979.and.yrARRq.le.2014 .and. monARRq.eq.7)
	timeINDq8 	= ind(yrARRq.ge.1979.and.yrARRq.le.2014 .and. monARRq.eq.8)

	qM  			= new((/3,12,181,360/),"float")
	qM(0,:,:,:)	= dim_avg_n_Wrap(q(timeINDq6,:,:,:),0)
	qM(1,:,:,:)	= dim_avg_n_Wrap(q(timeINDq7,:,:,:),0)
	qM(2,:,:,:)	= dim_avg_n_Wrap(q(timeINDq8,:,:,:),0)
	qMM 			= dim_avg_n_Wrap(qM,0) ;[lev]x[lat]x[lon]
	printVarSummary(qMM)

  qJJA = new((/3,36,12,181,360/),"float")
  qJJA(0,:,:,:,:)	= q(timeINDu6,:,:,:)
	qJJA(1,:,:,:,:)	= q(timeINDu7,:,:,:)
	qJJA(2,:,:,:,:)	= q(timeINDu8,:,:,:)


	dq 		= new(dimsizes(qJJA),"float")
  copy_VarCoords(qJJA, dq)
  do i =0,35
    dq(:,i,:,:,:) 		= qJJA(:,i,:,:,:) - qM
  end do
	dqM 		= dim_avg_n_Wrap(dq,0)

	;; 地面气压sp===========================================
	timeARRsp 	= cd_calendar(fsp->time,-5) ;读取时间的位置
	yrARRsp 		= timeARRsp(:,0)
	monARRsp		= timeARRsp(:,1)
	timeINDsp6 = ind(yrARRsp.ge.1979.and.yrARRsp.le.2014 .and. monARRsp.eq.6)
	timeINDsp7 = ind(yrARRsp.ge.1979.and.yrARRsp.le.2014 .and. monARRsp.eq.7)
	timeINDsp8 = ind(yrARRsp.ge.1979.and.yrARRsp.le.2014 .and. monARRsp.eq.8)

	spM  			= new((/3,181,360/),"float")
	; spM(0,:,:)	= dim_avg_n_Wrap(sp(timeINDsp10,:,:),0)
	spM(0,:,:)	= dim_avg_n_Wrap(sp(timeINDsp6,:,:),0)
	spM(1,:,:)	= dim_avg_n_Wrap(sp(timeINDsp7,:,:),0)
	spM(2,:,:)	= dim_avg_n_Wrap(sp(timeINDsp8,:,:),0)
	spMM 			= dim_avg_n_Wrap(spM,0) ;[lev]x[lat]x[lon]
	printVarSummary(spMM)

;;垂直水汽===============================================
	p 		= u&level	;[12]   以hPa为单位
	dqdp 	= center_finite_diff_n(dq,p,False,0,2) ;[month]x[years]x[lev]x[lat]x[lon]
	copy_VarCoords(dq,dqdp)
	printVarSummary(dqdp)
	; print(dqdp(0,0,:,0,0))

	Cdqdp	= center_finite_diff_n(qMM,p,False,0,0) ;[lev]x[lat]x[lon]
	copy_VarCoords(qMM,Cdqdp)
	printVarSummary(Cdqdp)
	; print(Cdqdp(:,0,0))

;;水平水汽===============================================
	dqdx 	= new(dimsizes(dq),"float")  ;[month]x[years]x[lev]x[lat]x[lon]
	dqdy 	= new(dimsizes(dq),"float")
	Cdqdx 	= new(dimsizes(qMM),"float") ;[lev]x[lat]x[lon]
	Cdqdy 	= new(dimsizes(qMM),"float")
	
	LAT 	= dq&latitude
	LON 	= dq&longitude
	dlon 	= (LON(2)-LON(1))*0.0174533
	dlat 	= (LAT(2)-LAT(1))*0.0174533
	do i = 0,dimsizes(LAT)-1
		dx 	= 6378388.*cos(0.0174533*LAT(i))*dlon
		dqdx(:,:,:,i,:) 	= center_finite_diff_n(dq(:,:,:,i,:),dx,False,0,3)
		Cdqdx(:,i,:) 	= center_finite_diff_n(qMM(:,i,:),dx,False,0,1)
	end do  
	copy_VarCoords(dq,dqdx)
	copy_VarCoords(qMM,Cdqdx)
	do i = 0,dimsizes(LON)-1
		dy 	= 6378388.*dlat
		dqdy(:,:,:,:,i) 	= center_finite_diff_n(dq(:,:,:,:,i),dy,False,0,3)
		Cdqdy(:,:,i) 	= center_finite_diff_n(qMM(:,:,i),dy,False,0,1)
	end do
	copy_VarCoords(dq,dqdy)
	copy_VarCoords(qMM,Cdqdx)
	printVarSummary(dqdx)
	print(dqdx(0,0,:,0,0))
	print(dqdy(0,0,:,0,0))

;;计算水汽收支的偏差方程各项===========================
	Te1= new(dimsizes(dq), "float")   ;;[month]x[year|36]x[lev|12]x[lat]x[lon]
	Te2=new(dimsizes(dq), "float")
	Te3=new(dimsizes(dq), "float")	 ;;[month]x[year|36]x[lev|12]x[lat]x[lon]
	Te4=new(dimsizes(dq), "float")
  do nyear = 0,35
    do i = 0,2
      Te1(i,nyear,:,:,:)=-1.0*wMM*dqdp(i,nyear,:,:,:)  ;垂直水汽输送热力项
      Te2(i,nyear,:,:,:)=-1.0*dw(i,nyear,:,:,:)*Cdqdp(:,:,:)  ;垂直水汽输送动力项
      Te3(i,nyear,:,:,:)=-1.0*uMM*dqdx(i,nyear,:,:,:)-1.0*vMM*dqdy(i,nyear,:,:,:)  ;水平水汽输送热力项
      Te4(i,nyear,:,:,:)=-1.0*du(i,nyear,:,:,:)*Cdqdx-1.0*dv(i,nyear,:,:,:)*Cdqdy  ;水平水汽输送动力项
    end do
  end do
	copy_VarCoords(dw,Te1)
	copy_VarCoords(dw,Te2)
	copy_VarCoords(dq,Te3)
	copy_VarCoords(dq,Te4)
	printVarSummary(Te1)	
	printVarSummary(Te3)
	; print(Te1(0,0,:,0,0))
	; print(Te2(0,0,:,0,0))
	; print(Te3(0,:,0,0))
	; print(Te4(0,0,:,0,0))

;;垂直积分===============================================
	ptop 	= min(p)
	dp 		= dpres_plevel(p, spM, ptop, 0) ;[month]x[lev]x[lat]x[lon]   hPa为单位
	printVarSummary(dp) 
	; print(dp(1,:,0,0))

	Tee1= new(dimsizes(dq), "float")   ;;[month]x[year|36]x[lev|12]x[lat]x[lon]
	Tee2=new(dimsizes(dq), "float")
	Tee3=new(dimsizes(dq), "float")	 ;;[month]x[year|36]x[lev|12]x[lat]x[lon]
	Tee4=new(dimsizes(dq), "float")


	do nyear=0,35
    Tee1(:,nyear,:,:,:)=Te1(:,nyear,:,:,:)*dp
    Tee2(:,nyear,:,:,:)=Te2(:,nyear,:,:,:)*dp
    Tee3(:,nyear,:,:,:)=Te3(:,nyear,:,:,:)*dp
    Tee4(:,nyear,:,:,:)=Te4(:,nyear,:,:,:)*dp
  end do
	T1=dim_sum_n_Wrap(Tee1,2) ;垂直积分，变为[month]x[year]x[lat]x[lon]
	T2=dim_sum_n_Wrap(Tee2,2)
	T3=dim_sum_n_Wrap(Tee3,2)
	T4=dim_sum_n_Wrap(Tee4,2)
	T1=T1/9.81*86400  ;;单位换算成mm/day
	T2=T2/9.81*86400
	T3=T3/9.81*86400*100
	T4=T4/9.81*86400*100

	copy_VarCoords(dw(:,:,0,:,:),T1)
	copy_VarCoords(dw(:,:,0,:,:),T2)
	copy_VarCoords(dq(:,:,0,:,:),T3)
	copy_VarCoords(dq(:,:,0,:,:),T4)

	T1draw = dim_avg_n_Wrap(T1,0) ;[year]x[lat]x[lon]
	T2draw = dim_avg_n_Wrap(T2,0)
	T3draw = dim_avg_n_Wrap(T3,0)
	T4draw = dim_avg_n_Wrap(T4,0)
	printMinMax(T1draw(0,{0:40},{95:150}),True) ;min=-2.85205   max=4.00643	
	printMinMax(T2draw(0,{0:40},{95:150}),True) ;min=-11.0172   max=18.0444
	printMinMax(T3draw(0,{0:40},{95:150}),True) ;min=-6.84386   max=5.86864
	printMinMax(T4draw(0,{0:40},{95:150}),True) ;min=-4.3223   max=4.08617

	output="/home/ys17-23/chenhj/0302code/"
   system("rm -f "+output+"ERA5_ncl_qbudget_zzy.nc")
   fout = addfile(output+"ERA5_ncl_qbudget_zzy.nc","c")
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