import { ExtractJwt, Strategy } from 'passport-jwt';
import { PassportStrategy } from '@nestjs/passport';
import { Inject, Injectable } from '@nestjs/common';
import access_tokenJwtConfig from '../@config/access_token-jwt.config';
import { ConfigType } from '@nestjs/config';
import refresh_tokenJwtConfig from '../@config/refresh_token-jwt.config';
import { Request } from 'express';
import 'dotenv/config';
import { AuthService } from '../auth.service';

@Injectable()
export class RefreshJwtStrategy extends PassportStrategy(
  Strategy,
  'refresh-jwt',
) {
  constructor(
    private authService: AuthService,
    @Inject(refresh_tokenJwtConfig.KEY)
    private refreshTokenJwtConfig: ConfigType<typeof access_tokenJwtConfig>,
  ) {
    super({
      jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
      ignoreExpiration: false,
      secretOrKey: refreshTokenJwtConfig.secret,
      passReqToCallback: true,
    });
  }

  async validate(payload: any, req: Request) {
    const authorizationHeader = req.headers['authorization'];
    if (!authorizationHeader) {
      throw new Error('Authorization header not found');
    }

    const refreshToken = authorizationHeader.replace('Bearer', '').trim();
    const userId = payload.sub;

    return this.authService.validateRefreshToken(userId, refreshToken);
  }
}
