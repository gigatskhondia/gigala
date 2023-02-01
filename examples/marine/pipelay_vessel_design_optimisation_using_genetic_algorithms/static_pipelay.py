# import math
import numpy as np


def static_pipe_lay(ODs,ts ,Es, SMYS,rho_s ,tFBE ,rhoFBE,tconc, rho_conc,
            d, rho_sea,
            mu_roller,
            thetaPT, LFL, RB ,ELPT, LPT, ELWL, LMP, RS, CL):

    Pi= 3.14
    As = Pi/4* (ODs**2 - (ODs - 2*ts)**2)
    # Is =Pi/64 * (ODs**4 - (ODs - 2*ts)**4)
    AFBE = Pi/4*((ODs + 2*tFBE)**2 - ODs**2)
    Aconc=Pi/4*((ODs+2*tFBE+2*tconc)**2-(ODs+2*tFBE)**2)
    Dhyd = ODs + 2*tFBE + 2*tconc
    Ms = As*rho_s/10**6
    g = 9.81
    Ws = Ms*g
    MFBE = AFBE*rhoFBE/10**6
    WFBE = MFBE*g
    Mconc = Aconc*rho_conc/10**6
    Wconc = Mconc*g
    Vsea = Pi/4* Dhyd**2
    # Msea = Vsea*rho_sea/10**6
    Wsea = Vsea*rho_sea*g/10**6
    Wair = Wconc + WFBE + Ws
    Wsub = Wconc + WFBE + Ws - Wsea
    xob = RB*np.sin(np.deg2rad(thetaPT))
    ELob = ELPT + RB - np.sqrt(RB**2 - xob**2)
    ELMP =ELob - RB + np.sqrt(RB**2-(LPT +xob -LMP )**2)
    thetaMP=np.arcsin((LPT + xob - LMP)/RB)
    thetaMP=np.rad2deg(thetaMP)
    xos = RS*np.sin(np.deg2rad(thetaMP))
    ELos=ELMP + RS - np.sqrt(RS**2-xos**2)
    thetaS = 2*np.arcsin(CL/(2*RS))
    thetaS=np.rad2deg(thetaS)
    thetaWL =np.arccos(1-(ELos-ELWL)/RS)
    thetaWL=np.rad2deg(thetaWL)

    thetaLO = thetaMP + thetaS
    hLO = RS*(1 - np.cos(np.deg2rad(thetaLO))) - (ELos - ELWL)
    TLO = Wsub*(d - hLO)/( 1 - np.cos(np.deg2rad(thetaLO)))
    H = TLO - Wsub*(d - hLO)
    # Ls = H/Wsub *np.tan(np.deg2rad(thetaLO))
    # LTD = H/Wsub *math.asinh(np.tan(np.deg2rad(thetaLO)))
    # RC = H/Wsub *math.cosh( Wsub*LTD/H)**2
    RTD = H/Wsub

    def T2(T1,phi1,phi2,mu,w,R):
        num = 2*T1+R*w*(mu*(np.sin(np.deg2rad(phi1))-np.sin(np.deg2rad(phi2)))-np.cos(np.deg2rad(phi1)) +
                np.cos(np.deg2rad(phi2)))-T1
    
        den = 1-mu*np.sin(np.deg2rad((phi1-phi2)/2))
 
        return num/den

    TWL = T2(TLO,thetaLO,thetaWL,mu_roller,Wsub,RS)
    TMP = T2(TWL,thetaWL,thetaMP,mu_roller,Wair,RS)
    TPT = T2(TMP,thetaMP,thetaPT,mu_roller,Wair,RB)
    Ttens = LFL*Wair*(np.sin(np.deg2rad(thetaPT)) + mu_roller*np.cos(np.deg2rad(thetaPT))) + TPT
    # print(Ttens/1000*0.1019716213)
    Fa = H
    sigmaTDa = Fa/As
    sigmaTDe=-(rho_sea*d*g)*Dhyd**2/(4*ODs*ts)
    sigmaTDb = Es/RTD*ODs/2
    sigmaTDh=-(rho_sea*d*g)*ODs/(2*ts)
    sigmaTDlt = sigmaTDa + sigmaTDb + sigmaTDe/1000000
    sigmaTDlb = sigmaTDa - sigmaTDb + sigmaTDe/1000000
    sigmaTD=max(np.sqrt(sigmaTDlt**2+(sigmaTDh/1000000)**2-(sigmaTDh/1000000)*sigmaTDlt),
        np.sqrt(sigmaTDlb**2+(sigmaTDh/1000000)**2-(sigmaTDh/1000000)*sigmaTDlb))
    # print(sigmaTD/SMYS)
    sigmaLOa = TLO/As
    sigmaLOe=-(rho_sea*hLO*g)*Dhyd**2/(4*ODs*ts)
    sigmaLOb = Es/RS*ODs/2
    sigmaLOh=-(rho_sea*hLO*g)*ODs/(2*ts)
    sigmaLOlt = sigmaLOa - sigmaLOb + sigmaLOe/1000000
    sigmaLOlb = sigmaLOa + sigmaLOb + sigmaLOe/1000000
    sigmaLO=max(np.sqrt(sigmaLOlt**2+(sigmaLOh/1000000)**2-(sigmaLOh/1000000)*sigmaLOlt),
        np.sqrt(sigmaLOlb**2+(sigmaLOh/1000000)**2-(sigmaLOh/1000000)*sigmaLOlb))
    sigmaWLa = TWL/As
    sigmaWLb = Es/RS * ODs/2
    sigmaWL = sigmaWLa + sigmaWLb
    # print(sigmaWL/SMYS)
    sigmaMPa = TMP/As
    sigmaMPb = Es/min(RS,RB) * ODs/2
    sigmaMP = sigmaMPa + sigmaMPb
    # print(sigmaMP/SMYS)
    sigmaPTa= TPT/As
    sigmaPTb = Es/RB * ODs/2
    sigmaPT = sigmaPTa+ sigmaPTb
    # print(sigmaPT/SMYS)

    # Total tensioner requirements to recover pipe onboard
    Ttens_tonnef = Ttens/1000*0.1019716213 # [tonnef]

    # Steel stress at touchdown, aim for < 60%
    TTS_ratio = sigmaTD/SMYS

    # Maximum steel stress at barge, aim for < 90%
    TopS_ratio = max(sigmaLO, sigmaWL, sigmaMP, sigmaPT)/SMYS

    return Ttens_tonnef, TTS_ratio, TopS_ratio


if __name__ == "__main__":
    # Pipe data
    ODs = 323.9  # Outer diameter of steel pipe, [mm]
    ts = 14.2   # Wall thickness of steel pipe, [mm]
    Es = 207  # Young's modulus of steel, [GPa]
    SMYS = 358  # SMYS for X52 steel, [MPa]
    rho_s = 7850   # Density of steel,[kg⋅m^−3]
    tFBE = 0.5  # Thickness of FBE insulation layer, [mm]
    rhoFBE = 1300  # Density of FBE, [kg⋅m^−3]
    tconc = 50   # Thickness of concrete coating,[mm]
    rho_conc = 2250   # Density of concrete,[kg⋅m^−3]

    # Environmental data
    d = 50   # Water depth, [m]
    rho_sea = 1025    # Density of seawater,[kg⋅m^−3]

    # Pipe Launch Rollers
    mu_roller = 0.1   # Roller friction for pipe on stinger.

    # Lay-Barge Input Data
    thetaPT = 3  # Angle of inclination of firing line from horizontal, [deg]
    LFL = 116  # Length of pipe on inclined firing line, [m]
    RB = 150   # Radius of over-bend curve  between stinger and straight section of firing line, [m]
    ELPT = 8   # Height of Point of Tangent above  Reference Point (sternpost at keel level),[m]
    LPT = 10  # Horizontal distance between Point of Tangent and Reference Point, [m]
    ELWL = 6  # Elevation of Water Level above Reference Point, [m]
    LMP = 5   # Horizontal distance between Reference Point and Marriage Point, [m]
    RS = 135  # Stinger radius, [m]
    CL = 60   # Chord length of the stinger between the Marriage Point and the
              # Lift-Off Point at the second from last roller , [m]

    Ttens_tonnef, TTS_ratio, TopS_ratio = static_pipe_lay(ODs, ts, Es, SMYS, rho_s, tFBE, rhoFBE, tconc, rho_conc,
             d, rho_sea,
             mu_roller,
             thetaPT, LFL, RB, ELPT, LPT, ELWL, LMP, RS, CL)

    print(Ttens_tonnef, TTS_ratio, TopS_ratio)