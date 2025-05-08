import { ExtractJwt, Strategy } from 'passport-jwt';
import { PassportStrategy } from '@nestjs/passport';
import { Inject, Injectable } from '@nestjs/common';
import 'dotenv/config';
import access_tokenJwtConfig from '../@config/access_token-jwt.config';
import { ConfigType } from '@nestjs/config';
import { AuthService } from '../auth.service';

@Injectable()
export class JwtStrategy extends PassportStrategy(Strategy) {
  constructor(
    private authService: AuthService,
    @Inject(access_tokenJwtConfig.KEY)
    private accessTokenJwtConfig: ConfigType<typeof access_tokenJwtConfig>,
  ) {
    super({
      jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
      ignoreExpiration: false,
      secretOrKey: accessTokenJwtConfig.secret,
    });
  }

  async validate(payload: any) {
    const data = this.authService.validateJWTUser(payload.sub);

    return data;
  }
}
