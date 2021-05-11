      Program driver
      double precision lngamma,x,p,q,psi(7),der(6),digama,trigam,
     #dble,err,log10,errapx
C
C             Program to test algorithm INBEDER.
C  
C             At the prompt, input values for p, q, and x; where p and q
C             are shape parameters and x is the value to which the beta
C             function is integrated.  The program outputs the incomplete
C             beta function and its first and second derivatives with
C             respect to p and/or q.
C
C             The program is intended to be run interactively.  It expects
C             that unit 5 is input and unit 6 is output.
C
      n=n2digit()
      write(6,1) n
 1    format(/,'Number of Bits for Fractional Component',/,
     #'of Floating Point Numbers is ',i4,/)
      j=nint(dble(n)*log10(2.d0))-4
      err=10.d0**(-j)
      write(6,2) err
 2    format(/,'The constant ERR should be set to ',d8.1,/)
      write(6,10)
 10   format(/,' To exit program, type Control-C',/)
 20   write(6,30)
 30   format(/,' Input p, q, and x')
      read(5,*,err=50) p,q,x
C
C             Compute the log of the complete beta function and
C             the derivatives of the log gamma functions
C
      psi(1)=lngamma(p,ier)+lngamma(q,ier)-lngamma(p+q,ier)
      psi(2)=digama(p,ier)
      psi(3)=trigam(p,ier)
      psi(4)=digama(q,ier)
      psi(5)=trigam(q,ier)
      psi(6)=digama(p+q,ier)
      psi(7)=trigam(p+q,ier)
C
C             Call subroutine INBEDER and print Results
C
      call inbeder(x,p,q,psi,der,nappx,errapx,ifault)
      write(6,40) nappx,errapx,ifault
 40   format(' Highest order approximant evaluated ='i4,/,
     #'Approximate maximum absolute error =',e15.8,', Ifault = ',i1)
      write(6,41) der
 41   format(' I =',e15.8, ',  Ip =',e15.8,', Ipp =',e15.8,/,
     #'Iq =',e15.8, ', Iqq =',e15.8,', Ipq =',e15.8)
      goto 20
 50   stop
      end





      integer function n2digit()
C
C             Function to compute the number of bits used to represent
C             the fractional component of floating point numbers.
C
      double precision one, two, D
      data one, two/1.d0, 2.d0/
      i=-1
 1    i=i+1
      d=(two**i+one/two)-(two**i-one/two)-one
      if ((-one/two.le.d).and.(d.le.one/two)) goto 1
      n2digit=i-1
      return
      end





      subroutine inbeder(x,p,q,psi,der,nappx,errapx,ifault)
C
C             x: Input argument -- value to which beta function is integrated
C             p,q: Input arguments -- beta shape parameters
C             psi: input arguments -- vector of length 7
C                                    psi(1) = log[Beta(p,q)]
C                                    psi(2) = digamma(p)
C                                    psi(3) = trigamma(p)
C                                    psi(4) = digamma(q)
C                                    psi(5) = trigamma(q)
C                                    psi(6) = digamma(p+q)
C                                    psi(7) = trigamma(p+q)
C             der: output -- vector of length 6
C                            der(1) = I (incomplete beta function)
C                            der(2) = Ip
C                            der(3) = Ipp
C                            der(4) = Iq
C                            der(5) = Iqq
C                            der(6) = Ipq
C             nappx: output -- highest order approximant evaluated
C             errapx: output -- approximate size of maximum absolute error
C                               of computed derivatives
C             ifault: output -- error indicator
C                               ifault = 0: no error
C                               ifault = 1: x outside of (0,1)
C                               ifault = 2: p less that 0
C                               ifault = 3: q less than 0
C                               ifault = 4: derivatives set to 0 because I=0
C                                           or I=1
C                               ifault = 5: evaluation stopped after maxappx
C                                           terms
C
      double precision x,p,q,psi(7),der(6),pa,pb,pab,pa1,pb1,pab1,
     #pp,qq,x1,c(6),dr(6),an1(6),an2(6),bn1(6),bn2(6),logx1,logomx,
     #w,omx,Rn,pr,an(6),bn(6),d,d1,dan(6),dbn(6),c0,zero,one,err,
     #log,exp,der_old(6),max,prmin,prmax,e,errapx
      data zero,one,err,maxappx,minappx,prmin/0.d0,1.d0,1.d-12,200,
     #3,1.d-24/
      prmax=one-err
C
C          Initialize derivative vectors 
C          and check for admissability of input arguments
C
      do 10 i=1,6
      der_old(i)=zero
      c(i)=zero
      dr(i)=one
      der(i)=zero
      an2(i)=zero
      bn2(i)=zero
      an1(i)=zero
 10   bn1(i)=zero
      an1(1)=one
      bn1(1)=one
      an2(1)=one
      pab=psi(6)
      pab1=psi(7)
      ifault=0
      nappx=0
      if ((x.le.zero).or.(x.ge.one)) then
      ifault=1
      return
      endif
      if (p.le.zero) then
      ifault=2
      return
      endif
      if (q.le.zero) then
      ifault=3
      return
      endif
C
C          Use I(x,p,q) = 1- I(1-x,q,p) if x > p/(p+q)
C
      if (x.le.p/(p+q)) then
      ii=1
      x1=x
      omx=one-x
      pp=p
      qq=q
      pa=psi(2)
      pb=psi(4)
      pa1=psi(3)
      pb1=psi(5)
      else
      ii=2
      x1=one-x
      omx=x
      pp=q
      qq=p 
      pa=psi(4)
      pb=psi(2)
      pa1=psi(5)
      pb1=psi(3)
      end if
      w=x1/omx
      logx1=log(x1)
      logomx=log(omx)
C
C          Compute derivatives of K(x,p,q) = x^p(1-x)^(q-1)/[p beta(p,q)]
C
      c(1)=pp*logx1+(qq-1)*logomx-psi(1)-log(pp)
      c0=exp(c(1))
      c(2)=logx1-one/pp-pa+pab
      c(3)=c(2)**2+one/pp**2-pa1+pab1
      c(4)=logomx-pb+pab
      c(5)=c(4)**2-pb1+pab1
      c(6)=c(2)*c(4)+pab1
C
C          Set counter and begin iteration
C
      n=0
 20   n=n+1
C
C          Compute derivatives of an and bn with respect to p and/or q
C
      call derconf(pp,qq,w,n,an,bn)
C
C          Use forward recurrance relations to compute An, Bn,
C          and their derivatives
C
      dan(1)=an(1)*an2(1)+bn(1)*an1(1)
      dbn(1)=an(1)*bn2(1)+bn(1)*bn1(1)
      dan(2)=an(2)*an2(1)+an(1)*an2(2)+bn(2)*an1(1)+bn(1)*an1(2)
      dbn(2)=an(2)*bn2(1)+an(1)*bn2(2)+bn(2)*bn1(1)+bn(1)*bn1(2)
      dan(3)=an(3)*an2(1)+2*an(2)*an2(2)+an(1)*an2(3)+bn(3)*an1(1)
     #     +2*bn(2)*an1(2)+bn(1)*an1(3)
      dbn(3)=an(3)*bn2(1)+2*an(2)*bn2(2)+an(1)*bn2(3)+bn(3)*bn1(1)
     #     +2*bn(2)*bn1(2)+bn(1)*bn1(3)
      dan(4)=an(4)*an2(1)+an(1)*an2(4)+bn(4)*an1(1)+bn(1)*an1(4)
      dbn(4)=an(4)*bn2(1)+an(1)*bn2(4)+bn(4)*bn1(1)+bn(1)*bn1(4)
      dan(5)=an(5)*an2(1)+2*an(4)*an2(4)+an(1)*an2(5)+bn(5)*an1(1)
     #     +2*bn(4)*an1(4)+bn(1)*an1(5)
      dbn(5)=an(5)*bn2(1)+2*an(4)*bn2(4)+an(1)*bn2(5)+bn(5)*bn1(1)
     #     +2*bn(4)*bn1(4)+bn(1)*bn1(5)
      dan(6)=an(6)*an2(1)+an(2)*an2(4)+an(4)*an2(2)+an(1)*an2(6)
     #     +bn(6)*an1(1)+bn(2)*an1(4)+bn(4)*an1(2)+bn(1)*an1(6)
      dbn(6)=an(6)*bn2(1)+an(2)*bn2(4)+an(4)*bn2(2)+an(1)*bn2(6)
     #     +bn(6)*bn1(1)+bn(2)*bn1(4)+bn(4)*bn1(2)+bn(1)*bn1(6)
C
C          Scale derivatives to prevent overflow
C
      Rn=dan(1)
      iii=1
      if (abs(dbn(1)).gt.abs(dan(1))) then
      Rn=dbn(1)
      iii=2
      endif
      do 33 i=1,6
      an1(i)=an1(i)/Rn
 33   bn1(i)=bn1(i)/Rn
      do 34 i=2,6
      dan(i)=dan(i)/Rn
 34   dbn(i)=dbn(i)/Rn
      if (iii.eq.1) then
      dbn(1)=dbn(1)/dan(1)
      dan(1)=one
      else
      dan(1)=dan(1)/dbn(1)
      dbn(1)=one
      endif
C
C          Compute components of derivatives of the nth approximant
C
      dr(1)=dan(1)/dbn(1)
      Rn = dr(1)
      dr(2) = (dan(2)-Rn*dbn(2))/dbn(1)
      dr(3) = (-2*dan(2)*dbn(2)+2*Rn*dbn(2)**2)/dbn(1)**2+(dan(3)-Rn*dbn
     #(3))/dbn(1)
      dr(4) = (dan(4)-Rn*dbn(4))/dbn(1)
      dr(5) = (-2*dan(4)*dbn(4)+2*Rn*dbn(4)**2)/dbn(1)**2+(dan(5)-Rn*dbn
     #(5))/dbn(1)
      dr(6) = (-dan(2)*dbn(4)-dan(4)*dbn(2)+2*Rn*dbn(2)*dbn(4))/dbn(1)**
     #2+(dan(6)-Rn*dbn(6))/dbn(1)
C
C          Save terms corresponding to approximants n-1 and n-2
C
      do 30 i=1,6
      an2(i)=an1(i)
      an1(i)=dan(i)
      bn2(i)=bn1(i)
 30   bn1(i)=dbn(i)
C
C          Check if I < prmin or I > prmax
C
      if (dr(1).gt.zero) then
      pr=exp(c(1)+log(dr(1)))
      else
      pr=zero
      endif
      der(1)=pr
      if ((pr.lt.prmin).or.(pr.gt.prmax)) then
      errapx = abs(der_old(1)-pr)
      if (errapx.le.err) then
      ifault=4
      do 72 i=2,6
 72   der(i)=zero
      goto 75
      endif
      endif
C
C          Compute nth approximants
C
      der(2)=pr*c(2)+c0*dr(2)
      der(3)=pr*c(3)+2*c0*c(2)*dr(2)+c0*dr(3)
      der(4)=pr*c(4)+c0*dr(4)
      der(5)=pr*c(5)+2*c0*c(4)*dr(4)+c0*dr(5)
      der(6)=pr*c(6)+c0*c(4)*dr(2)+c0*c(2)*dr(4)+c0*dr(6)
C
C          Check for convergence, check for maximum and minimum iterations.
C          
      d=zero
      errapx=zero
      do 92 i=1,6
      d1=max(err,abs(der(i)))
      e=abs(der_old(i)-der(i))
      d1=e/d1
      if (d1.gt.d) d=d1
      if (e.gt.errapx) errapx=e
 92   der_old(i)=der(i)
      if (n.lt.minappx) d=one
      if (n.ge.maxappx) then
      d=zero
      ifault=5
      endif
      if (d.ge.err) goto 20
 75   continue
C
C          Adjust results if I(x,p,q) = 1- I(1-x,q,p) was used
C
      if (ii.eq.2) then
      der(1)=one-der(1)
      c0=der(2)
      der(2)=-der(4)
      der(4)=-c0
      c0=der(3)
      der(3)=-der(5)
      der(5)=-c0
      der(6)=-der(6)
      end if
      nappx=n
      return
      end





      subroutine derconf(p,q,w,n,an,bn)
C
C          Compute derivatives of an and bn with respect to p and/or q
C
      implicit double precision(t)
      double precision p,q,w,F,an(6),bn(6),zero,one
      data zero,one,two/0.d0,1.d0,2.d0/
      F = w*q/p
      if (n.eq.1) then
      t1=one-one/(p+one)
      t2=one-one/q
      t3=one-two/(p+two)
      t4=one-two/q
      an(1)=t1*t2*F
      an(2)=-an(1)/(p+one)
      an(3)=-two*an(2)/(p+one)
      an(4)=t1*F/q
      an(5)=zero
      an(6)=-an(4)/(p+one)
      bn(1)=one-t3*t4*F
      bn(2)=t3*t4*F/(p+two)
      bn(3)=-two*bn(2)/(p+two)
      bn(4)=-t3*F/q
      bn(5)=zero
      bn(6)=-bn(4)/(p+two)
      else
      call subd(n,p,q,F,an,bn)
      end if
      return
      end





      subroutine subd(in,p,q,F,an,bn)
C
C          Compute derivatives of an and bn with respect to p and/or q
C          when n > 1
C
      implicit double precision(t,c)
      double precision p,q,F,an(6),bn(6),n,dble
      data c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c16,c18,
     #c19,c20,c22,c24,c25,c26,c27,c28,c32,c35,c36,c38,c40,c44,c48,c50,
     #c51,c52,c53,c54,c60,c64,c65,c69,c70,c72,c77,c80,c87,c88,c96,c104,
     #c128,c130,c144,c155,c192,c224,c240,c288/0.d0,1.d0,2.d0,3.d0,4.d0,
     #5.d0,6.d0,7.d0,8.d0,9.d0,10.d0,11.d0,12.d0,13.d0,14.d0,16.d0,
     #18.d0,19.d0,20.d0,22.d0,24.d0,25.d0,26.d0,27.d0,28.d0,32.d0,35.d0,
     #36.d0,38.d0,40.d0,44.d0,48.d0,50.d0,51.d0,52.d0,53.d0,54.d0,60.d0,
     #64.d0,65.d0,69.d0,70.d0,72.d0,77.d0,80.d0,87.d0,88.d0,96.d0,
     #104.d0,128.d0,130.d0,144.d0,155.d0,192.d0,224.d0,240.d0,288.d0/
      n=dble(in)
      t2 = F**2
      t3 = C2*n-C2
      t5 = p*q
      t7 = C1/(t3*q+t5)
      t8 = t2*t7
      t9 = n**2
      t10 = t9**2
      t11 = t2*t10
      t12 = C4*n-C2
      t13 = q**2
      t14 = t12*t13
      t15 = p*t13
      t17 = C1/(t14+C2*t15)
      t19 = t9*n
      t20 = t19*t2
      t22 = C1/(p+C2*n-C1)
      t23 = t20*t22
      t24 = C2*n-C1
      t27 = C1/(t24*q+t5)
      t28 = t20*t27
      t30 = t10*n*t2
      t32 = n*t2
      t33 = C2*n-C3
      t36 = C1/(t33*t13+t15)
      t37 = t32*t36
      t38 = t9*t2
      t39 = C1/t13
      t41 = t32*t39
      t43 = (-C8+C4*n)*n
      t47 = C1/(C4+t43+(C4*n-C4+p)*p)
      t49 = t38*t17
      t50 = t38*t47
      t51 = t20*t47
      t52 = C1/q
      t54 = t2*t47
      t55 = t32*t47
      t57 = C1/(C2*p+C4*n-C6)
      t59 = C4*t8-C3*t11*t17-C4*t23-t28-C4*t30*t27+C9*t37-t38*
     #t39+t41+C4*t11*t47-t49+C24*t50-C16*t51-t2*t52+C4*t54-C16
     #*t55-C53*t38*t57
      t62 = C1/(p+C2*n-C2)
      t63 = t32*t62
      t65 = C1/(C2*p+C4*n-C2)
      t69 = t2/(p+C2*n-C3)
      t70 = t69*t19
      t73 = C1/(t3*t13+t15)
      t74 = t11*t73
      t76 = t10*t9*t2
      t79 = C1/(t24*t13+t15)
      t81 = t2*t62
      t82 = C4+t43
      t84 = C4*n-C4
      t89 = C1/(t82*t13+(t84*t13+t15)*p)
      t91 = t20*t36
      t92 = t11*t27
      t96 = t20*t89
      t97 = t20*t7
      t98 = t12*q
      t100 = C1/(t98+C2*t5)
      t102 = C51*t32*t57-C24*t63+C5*t38*t65+C12*t70+C40*t74+C2
     #*t76*t79+C8*t81+C4*t76*t89+C52*t91+C6*t92-C2*t69*t10-C8
     #*t20*t62+C2*t11*t22-C16*t96-C64*t97+t32*t100
      t104 = t38*t62
      t105 = t30*t36
      t107 = C4*n-C6
      t108 = t107*q
      t110 = C1/(t108+C2*t5)
      t113 = t38*t73
      t116 = C1/(t33*q+t5)
      t117 = t11*t116
      t118 = t20*t116
      t119 = t30*t79
      t120 = t32*t73
      t122 = t20*t73
      t123 = t20*t79
      t126 = C24*t104+C14*t105+t32*t52+C87*t32*t110-C9*t69-C12
     #*t30*t73+C24*t113-C26*t117+C65*t118-C2*t119-C4*t120+C4
     #*t30*t116-C48*t122+C2*t123-C2*t76*t36-C3*t38*t100
      t132 = C1/(t82*q+(t84*q+t5)*p)
      t133 = t20*t132
      t135 = t38*t89
      t136 = t11*t89
      t137 = t30*t89
      t138 = t11*t132
      t139 = t107*t13
      t141 = C1/(t139+C2*t15)
      t142 = t38*t141
      t143 = t32*t132
      t144 = t32*t7
      t145 = t38*t7
      t149 = t38*t132
      t151 = t2*t116
      t152 = -C48*t133-C8*t30*t132+C4*t135+C24*t136-C16*t137+c32
     #*t138-C69*t142-C8*t143-C32*t144+C72*t145-t32*t65+C20
     #*t11*t7-C77*t11*t141+C32*t149-C155*t38*t110-C9*t151
      t155 = t84*n
      t156 = C1+t155
      t161 = C1/(t156*t13+(t14+t15)*p)
      t162 = t30*t161
      t163 = -C8+C8*n
      t164 = t163*n
      t165 = C2+t164
      t167 = -C4+C8*n
      t172 = C1/(t165*t13+(t167*t13+C2*t15)*p)
      t175 = (-C24+C8*n)*n
      t179 = C1/(C18+t175+(-C12+C8*n+C2*p)*p)
      t181 = t20*t161
      t182 = t38*t22
      t184 = (C24+t175)*n
      t186 = (-C24+C12*n)*n
      t192 = C1/(-C8+t184+(C12+t186+(-C6+C6*n+p)*p)*p)
      t198 = C1/(t156*q+(t98+t5)*p)
      t199 = t11*t198
      t200 = t20*t192
      t201 = -C4*t8+C2*t162+C3*t11*t172-C51*t32*t179+C2*t23+C4
     #*t28-C2*t181-C3*t182-C8*t11*t192-C6*t199+C32*t200-C6
     #*t37
      t207 = C1/(t165*q+(t167*q+C2*t5)*p)
      t210 = (-C12+C4*n)*n
      t211 = C9+t210
      t216 = C1/(t211*t13+(t139+t15)*p)
      t217 = t32*t216
      t218 = -C8+t184
      t220 = C12+t186
      t222 = -C6+C6*n
      t229 = C1/(t218*t13+(t220*t13+(t222*t13+t15)*p)*p)
      t230 = t11*t229
      t231 = t20*t216
      t232 = t69*n
      t233 = t30*t216
      t234 = C18+t175
      t236 = -C12+C8*n
      t241 = C1/(t234*t13+(t236*t13+C2*t15)*p)
      t242 = t38*t241
      t243 = C3*t38*t207-C36*t50+C12*t51-C12*t54-C9*t217+C36
     #*t55+C12*t63-C48*t230-C52*t231-C13*t232-C14*t233+C69*t
     #242
      t245 = t32*t192
      t251 = C1/(t234*q+(t236*q+C2*t5)*p)
      t256 = C1/(C1+t155+(C4*n-C2+p)*p)
      t257 = t20*t256
      t258 = C32*t245-C2*t70-C10*t74-C6*t81-C22*t91-C4*t92+c60
     #*t96+C16*t97-C6*t104-C87*t32*t251-C2*t105+C4*t257
      t267 = C1/(t218*q+(t220*q+(t222*q+t5)*p)*p)
      t268 = t11*t267
      t269 = t11*t79
      t270 = t30*t229
      t271 = t32*t267
      t272 = C6*t69-C64*t268-C18*t113+C4*t117-C20*t118-t269+C32
     #*t270+C2*t119+C4*t120+C24*t122-C2*t123+C16*t271
      t276 = t32*t27
      t277 = t69*t9
      t278 = t38*t116
      t279 = t38*t192
      t281 = C77*t11*t241-t276+C88*t133-C28*t135-C52*t136+C16*
     #t137+C9*t277+C35*t278-C28*t138-C48*t279+C40*t143+C155*
     #t38*t251
      t286 = C1/(t211*q+(t108+t5)*p)
      t287 = t20*t286
      t288 = t2*t192
      t292 = C1/(C9+t210+(C4*n-C6+p)*p)
      t293 = t2*t292
      t294 = t2*t286
      t295 = t20*t267
      t296 = t2*t132
      t297 = t32*t89
      t299 = C24*t144-C36*t145-C96*t149-C65*t287+C6*t151-C8*
     #t288+C9*t293+C9*t294+C96*t295-C4*t296+C4*t297-C4*t30*t
     #286
      t304 = t11*t286
      t305 = t32*t116
      t308 = t38*t267
      t309 = t11*t36
      t311 = t38*t79
      t315 = C1/(C2+t164+(-C4+C8*n+C2*p)*p)
      t317 = C2*t11*t292-t32*t207-C2*t11*t256+C26*t304-C25*t305+
     #C4*t30*t198+C16*t30*t267-C64*t308+C11*t309-C8*t76*t229+t
     #311-C5*t38*t315
      t319 = t32*t22
      t320 = t20*t198
      t321 = t20*t292
      t322 = t38*t229
      t323 = t38*t27
      t324 = t20*t229
      t328 = t38*t36
      t329 = t38*t172
      t330 = t32*t315+t319+t320-C12*t321-C8*t322+t323+C32*t324-C2
     #*t76*t161+C2*t76*t216+C53*t38*t179+C19*t328+t329
      t336 = (C6+t236*n)*n
      t337 = -C1+t336
      t340 = (-C12+C12*n)*n
      t341 = C3+t340
      t343 = -C3+C6*n
      t350 = C1/(t337*t13+(t341*t13+(t343*t13+t15)*p)*p)
      t357 = (-C64+(C96+(-C64+C16*n)*n)*n)*n
      t358 = C16+t357
      t363 = (C96+(-C96+C32*n)*n)*n
      t364 = -C32+t363
      t367 = (-C48+C24*n)*n
      t368 = C24+t367
      t378 = C1/(t358*q+(t364*q+(t368*q+(t163*q+t5)*p)*p)*p)
      t383 = (C54+(-C36+C8*n)*n)*n
      t384 = -C27+t383
      t387 = (-C36+C12*n)*n
      t388 = C27+t387
      t390 = -C9+C6*n
      t397 = C1/(t384*q+(t388*q+(t390*q+t5)*p)*p)
      t410 = C1/(t358*t13+(t364*t13+(t368*t13+(t163*t13+t15)*p)*p)*p)
      t413 = t32*t286
      t414 = -C3*t11*t350+C2*t8-C4*t162-C2*t28+C4*t181+t182+C8
     #*t199-C288*t20*t378-C32*t200+C2*t37+C8*t30*t397+C24*t3
     #8*t410+C4*t76*t350+C50*t413+C14*t50
      t422 = C1/(C16+t357+(-C32+t363+(C24+t367+(-C8+C8*n+p)*
     #p)*p)*p)
      t425 = t32*t229
      t426 = -C96*t20*t422+C14*t54+C12*t217-C28*t55-C96*t30*t4
     #10-C2*t63+C128*t230+C44*t231+C3*t232+C4*t233-C96*t245-
     #C8*t425+C2*t81+C4*t91-C52*t96
      t436 = C1/(t337*q+(t341*q+(t343*q+t5)*p)*p)
      t440 = C12*t11*t436-C4*t257-C2*t69+C72*t268+C6*t113-C18
     #*t38*t292+C2*t118+t269+C144*t11*t410-C40*t270-C2*t120-C4
     #*t122-C96*t271+t276-C36*t133
      t449 = C1/(t384*t13+(t388*t13+(t390*t13+t15)*p)*p)
      t456 = C1/(-C1+t336+(C3+t340+(-C3+C6*n+p)*p)*p)
      t458 = C38*t135+C22*t136-t277-C69*t38*t449-C7*t278+C96*t
     #279-C52*t143-C8*t144+C6*t145+C80*t149+C40*t287-C2*t151
     #+C32*t288-C12*t293+C4*t11*t456-C12*t294
      t468 = -C224*t295+C8*t296-C8*t297+C24*t76*t410-C8*t304-C52
     #*t11*t397+C7*t305+C87*t32*t397+C240*t308-t309-t311+c104
     #*t20*t449-C8*t20*t456+C5*t38*t456+t32*t436
      t469 = t38*t198
      t477 = -C2*t469+C26*t32*t292-t319-C8*t320+C4*t321+C64*t3
     #22+t323-C144*t324-C5*t328-C18*t2*t397+C24*t2*t422+C130*t
     #20*t397-C155*t38*t397-C96*t32*t422-C96*t20*t410
      t479 = t11*t161
      t485 = C1/(-C27+t383+(C27+t387+(C6*n-C9+p)*p)*p)
      t489 = t32*t198
      t492 = t2*t267
      t500 = C2*t479+C51*t32*t485-C77*t11*t449+C28*t30*t449+C2
     #*t489+C18*t32*t449-C2*t38*t161+C8*t492-t38*t350+C4*t20*t35
     #0-C8*t30*t436-C4*t30*t350+C6*t38*t256-C3*t38*t436+C24*t1
     #1*t422
      t507 = t38*t286
      t512 = t11*t216
      t517 = -C2*t32*t256-C4*t11*t485-C53*t38*t485+C144*t38*t422
     #+C24*t20*t485-C18*t2*t485-C70*t507-t32*t456-C48*t32*t378-C48
     #*t30*t378+C192*t38*t378-C22*t512+C192*t11*t378-C38*t38
     #*t216-C2*t20*t436-C4*t76*t449
      t521 = C16*t8-C8*t28+t41-C3*t49+C20*t74+C65*t91+C4*t92
     #-C48*t96-C16*t97+C4*t105+C72*t113-C4*t117+C24*t118+c6
     #*t269-C4*t119-C32*t120-C64*t122-t123-t276-C32*t133
      t526 = t2*t73
      t527 = t2*t36
      t528 = C48*t149-C18*t151+C8*t296-C8*t297+C51*t305-C26*
     #t309+C5*t323+t32*t17+C87*t32*t141+C4*t526-C9*t527
      t531 = t2*t89
      t532 = t32*t79
      t533 = -C32*t96+C8*t136+C48*t135-C48*t120+C24*t91+C48*
     #t113-C16*t122-C53*t328-C18*t527-C8*t123+C16*t526+C8*t5
     #31+C5*t311-t532+C51*t37-C4*t309+C4*t269-C32*t297
      t537 = C9*t2*t216-C87*t32*t241-t32*t172-C12*t8+C4*t162+C4
     #*t28+t181-C4*t199-C25*t37-C51*t413-C64*t230-C65*t231-C4
     #*t233+C155*t242+C16*t425
      t538 = -C20*t91+C88*t96-C16*t268-C36*t113-C4*t118-C4*t
     #269+C16*t270+C24*t120+C16*t122+C4*t123+C64*t271+C2*t27
     #6+C24*t133-C96*t135-C28*t136+C18*t278
      t540 = C72*t143+C24*t144-C12*t145-C72*t149-C24*t287+C12
     #*t151+C18*t294+C64*t295-C24*t296+C40*t297+C4*t304-C26
     #*t305-C96*t308+C4*t309+t311
      t541 = -C5*t469+C8*t320-C64*t322-C6*t323+C96*t324+C35*
     #t328+C3*t329-C6*t479+t489-C16*t492+C53*t507+C26*t512-C4
     #*t526+C6*t527-t532-C4*t531
      t544 = t9*F
      t546 = C1/(p+C2*n)
      t548 = q*n
      t550 = C1/(t5+C2*t548)
      t551 = t544*t550
      t552 = t544*t7
      t553 = n*F
      t554 = t553*t7
      t555 = t19*F
      t557 = F*t62
      t559 = t557*n
      t562 = C1-F+C2*t544*t546-C2*t551-C4*t552+C2*t554+C2*t5
     #55*t7-C2*t557-C2*t557*t9+C4*t559-C2*t555*t550+C2*t553*t5
     #2
      t563 = t553*t550
      t564 = t553*t132
      t567 = t544*t132
      t568 = F*t47
      t572 = C1/(C4*t9+(C4*n+p)*p)
      t574 = q*t9
      t578 = C1/(C4*t574+(C4*t548+t5)*p)
      t580 = t544*t578
      t582 = t553*t47
      t583 = -t563-C2*t564+C2*t544*t47-C2*t555*t132+C4*t567+C2
     #*t568-C2*t544*t572+C2*t555*t578-t551+C2*t580+t552-t554+t557-
     #t559+t553*t546-C4*t582
      t598 = C1/(C8*q*t19+(C12*t574+(C6*t548+t5)*p)*p)
      t608 = C2*t564-C2*t567-C2*t568+C2*t580+C2*t553*t578+C4
     #*t544/(C8*t19+(C12*t9+(C6*n+p)*p)*p)-C4*t555*t598-C2*t55
     #3*t572+C8*t553*t192-C4*F*t192-C4*t544*t192+C4*t553*t267-C8
     #*t544*t267+C4*t555*t267-C4*t544*t598+C2*t582
      t610 = F*t7
      an(1) = t59+t102+t126+t152
      an(2) = t201+t243+t258+t272+t281+t299+t317+t330
      an(3) = t414+t426+t440+t458+t468+t477+t500+t517
      an(4) = t521+C32*t135+C32*t136-C8*t137-t2*t39-C53*t278+C8
     #*t138-C155*t142-C32*t143-C48*t144+C48*t145+t528
      an(5) = t533
      an(6) = t537+t538+t540+t541
      bn(1) = t562
      bn(2) = t583
      bn(3) = t608
      bn(4) = -F*t52-C2*t552+C4*t554-C2*t610+C2*t551
      bn(5) = C0
      bn(6) = C2*t567-t554-C4*t564+t563-C2*t580+t610+C2*F*t132
      return
      end





        double precision function lngamma(z, ier)
c
c       Uses Lanczos-type approximation to ln(gamma) for z > 0.
c       Reference:
c            Lanczos, C. 'A precision approximation of the gamma
c                    function', J. SIAM Numer. anal., B, 1, 86-96, 1964.
c       Accuracy: About 14 significant digits except for small regions
c                 in the vicinity of 1 and 2.
c
c       Programmer: Alan Miller
c                   CSIRO Division of Mathematics & Statistics
c
c       N.B. It is assumed that the Fortran compiler supports long
c            variable names, including the underline character.   Some
c            compilers will not accept the 'implicit none' statement
c            below.
c
c       Latest revision - 17 April 1988
c
        implicit none
        double precision a(9), z, lnsqrt2pi, tmp
        double precision zero,half,one,sixp5,seven,log
        integer ier, j
        data zero,half,one,sixp5,seven/0.d0,5.d-1,1.d0,6.5d0,7.d0/
        data a/0.9999999999995183d0, 676.5203681218835d0,
     +         -1259.139216722289d0, 771.3234287757674d0,
     +         -176.6150291498386d0, 12.50734324009056d0,
     +         -0.1385710331296526d0, 0.9934937113930748d-05,
     +         0.1659470187408462d-06/
        data lnsqrt2pi/0.91893 85332 04672 7d0/
        lngamma = zero
        if (z .le. zero) then
          ier = 1
          return
        end if
        ier = 0
        tmp = z + seven
        do 10 j = 9, 2, -1
          lngamma = lngamma + a(j)/tmp
          tmp = tmp - one
   10   continue
        lngamma = lngamma + a(1)
        lngamma = log(lngamma) + lnsqrt2pi - (z+sixp5) +
     +                               (z-half)*log(z+sixp5)
        end





      DOUBLE PRECISION FUNCTION DIGAMA(X, IFAULT)
C
C     ALGORITHM AS 103  APPL. STATIST. (1976) VOL.25, NO.3
C
C     Calculates DIGAMMA(X) = D( LOG( GAMMA(X))) / DX
C
      DOUBLE PRECISION ZERO, HALF, ONE, X, Y, R, log
C
C     Set constants, SN = Nth Stirling coefficient, D1 = DIGAMMA(1.0)
C
      DATA ZERO/0.d0/, HALF/0.5d0/, ONE/1.d0/
      DATA S, C, S3, S4, S5, D1 /1.d-05, 8.5d0, 8.333333333d-02,
     *    8.3333333333d-03, 3.96825 3968d-03, -0.57721 56649/
C
C     Check argument is positive
C
      DIGAMA = ZERO
      Y = X
      IFAULT = 1
      IF (Y .LE. ZERO) RETURN
      IFAULT = 0
C
C     Use approximation if argument <= S
C
      IF (Y .LE. S) THEN
        DIGAMA = D1 - ONE / Y
        RETURN
      END IF
C
C     Reduce to DIGAMA(X + N) where (X + N) >= C
C
    1 IF (Y .GE. C) GO TO 2
      DIGAMA = DIGAMA - ONE/Y
      Y = Y + ONE
      GO TO 1
C
C     Use Stirling's (actually de Moivre's) expansion if argument > C
C
    2 R = ONE / Y
      DIGAMA = DIGAMA + LOG(Y) - HALF*R
      R = R * R
      DIGAMA = DIGAMA - R*(S3 - R*(S4 - R*S5))
      RETURN
      END





      double precision function trigam(x, ifault)
      implicit double precision (a-h,o-z) 
c
c        algorithm as121   Appl. Statist. (1978) vol 27, no. 1
c
c        calculates trigamma(x) = d**2(log(gamma(x))) / dx**2
c
      double precision a, b, one, half, b2, b4, b6,b8, x, y, z, zero
      data a, b, one, half /1.0d-4, 40.0d0, 1.0d0, 0.5d0/
      data zero /0.0d0/
c
c        b2, b4, b6 and b8 are Bernoulli numbers
c
      data b2, b4, b6,b8
     */0.1666666667d0, -0.03333333333d0, 0.02380952381, -0.03333333333/
c
c        check for positive value of x
c
      trigam = zero
      ifault = 1
      if (x.le.zero) return
      ifault = 0
      z = x
c
c        use small value approximation if x .le. a
c
      if (z .gt. a) goto 10
      trigam = one / (z * z)
      return
c
c        increase argument to (x+i) .ge. b
c
   10 if (z .ge. b) goto 20
      trigam = trigam + one / (z * z)
      z = z + one
      goto 10
c
c        apply asymptotic formula if argument .ge. b
c
   20 y = one / (z * z)
      trigam = trigam + half * y +
     * (one + y * (b2 + y * (b4 + y * (b6 + y * b8)))) / z
      return
      end
