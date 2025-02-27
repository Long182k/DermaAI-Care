import {
  Inject,
  Injectable,
  UnauthorizedException,
  NotFoundException,
} from '@nestjs/common';
import { UsersService } from '../users/users.service';
import { JwtService } from '@nestjs/jwt';
import refreshTokenJwtConfig from './@config/refresh_token-jwt.config';
import { ConfigType } from '@nestjs/config';
import * as argon from 'argon2';
import { UpdateHashedRefreshTokenDTO } from 'src/users/dto/update-user.dto';
import { CreateUserDTO } from 'src/users/dto/create-user.dto';
import { UserRepository } from 'src/users/users.repository';
import { PrismaService } from 'src/prisma.service';
import { ChangePasswordDto } from './dto/change-password.dto';
import { MailerService } from '@nestjs-modules/mailer';

@Injectable()
export class AuthService {
  constructor(
    private usersService: UsersService,
    private jwtService: JwtService,
    private userRepository: UserRepository,
    @Inject(refreshTokenJwtConfig.KEY)
    private refreshTokenConfig: ConfigType<typeof refreshTokenJwtConfig>,
    private prisma: PrismaService,
    private readonly mailerService: MailerService,
  ) {}

  async validateUser(email: string, password: string): Promise<any> {
    const user = await this.usersService.findOne(email);

    if (!user.isActive) {
      throw new UnauthorizedException('User is not active');
    }

    const isVerifiedPassword = await argon.verify(
      user.hashedPassword,
      password,
    );

    if (user && isVerifiedPassword) {
      const { hashedPassword, ...result } = user;
      return result;
    }

    return null;
  }

  async login(user: any) {
    const { accessToken, refreshToken } = await this.generateTokens(user);
    const hashedRefreshToke = await argon.hash(refreshToken);
    const payloadUpdate: UpdateHashedRefreshTokenDTO = {
      userId: user.id,
      hashedRefreshToken: hashedRefreshToke,
    };

    await this.usersService.updateHashedRefreshToken(payloadUpdate);

    return {
      accessToken,
      refreshToken,
      userId: user.id,
      userName: user.userName,
      email: user.email,
      role: user.role,
      avatarUrl: user.avatarUrl,
      coverPageUrl: user.coverPageUrl,
    };
  }

  async createUser(createUserDto: CreateUserDTO) {
    const { accessToken, refreshToken } =
      await this.generateTokens(createUserDto);

    const result = await this.userRepository.createUser(createUserDto);

    return {
      ...result,
      accessToken,
      refreshToken,
    };
  }

  async generateTokens(user: any) {
    const payload = {
      userId: user.id,
      userName: user.userName,
      email: user.email,
      role: user.role,
    };

    const [accessToken, refreshToken] = await Promise.all([
      this.jwtService.signAsync(payload),
      this.jwtService.signAsync(payload, this.refreshTokenConfig),
    ]);

    return {
      accessToken,
      refreshToken,
    };
  }

  async refreshToken(user: any) {
    const { accessToken, refreshToken } = await this.generateTokens(user);

    const hashedRefreshToke = await argon.hash(refreshToken);
    const payloadUpdate: UpdateHashedRefreshTokenDTO = {
      userId: user.id,
      hashedRefreshToken: hashedRefreshToke,
    };

    return {
      id: user.id,
      accessToken,
      refreshToken,
    };
  }

  async validateRefreshToken(userId: string, refreshToken: string) {
    const user = await this.usersService.findUserByKeyword({ id: userId });

    if (!user || !user.hashedRefreshToken) {
      throw new UnauthorizedException('Invalid Refresh Token');
    }

    const isRefreshTokenMatched = await argon.verify(
      user.hashedRefreshToken,
      refreshToken,
    );

    if (!isRefreshTokenMatched) {
      throw new UnauthorizedException('Invalid Refresh Token');
    }

    return { id: userId };
  }

  async validateJWTUser(userId: string) {
    const user = await this.usersService.findUserByKeyword({ id: userId });

    if (!user || !user.hashedRefreshToken) {
      throw new UnauthorizedException('User Not Found');
    }

    return {
      userId: user.id,
      userName: user.userName,
      role: user.role,
      email: user.email,
    };
  }

  async signOut(userId: string) {
    const payloadUpdate: UpdateHashedRefreshTokenDTO = {
      userId,
      hashedRefreshToken: null,
    };

    await this.prisma.user.update({
      where: { id: userId },
      data: {
        hashedRefreshToken: null,
        lastLoginAt: new Date(),
      },
    });

    return await this.usersService.updateHashedRefreshToken(payloadUpdate);
  }

  async getUserById(id: string) {
    return await this.usersService.findUserByKeyword({ id });
  }

  async changePassword(userId: string, changePasswordDto: ChangePasswordDto) {
    const user = await this.prisma.user.findUnique({
      where: { id: userId },
    });

    const isPasswordValid = await argon.verify(
      user.hashedPassword,
      changePasswordDto.oldPassword,
    );

    if (!isPasswordValid) {
      throw new UnauthorizedException('Current password is incorrect');
    }

    const hashedPassword = await argon.hash(changePasswordDto.newPassword);

    return this.prisma.user.update({
      where: { id: userId },
      data: { hashedPassword },
    });
  }

  async forgotPassword(email: string) {
    const user = await this.usersService.findUserByKeyword({ email });
    if (!user) {
      throw new NotFoundException('User with this email does not exist');
    }
    const newPassword = Math.random().toString(36).slice(-6);

    const hashedPassword = await this.hashPassword(newPassword);

    await this.prisma.user.update({
      where: { email },
      data: { hashedPassword },
    });

    await this.mailerService.sendMail({
      to: email,
      subject: 'Friendzii Social Media Platform - Password Reset',
      template: 'forgot-password',
      context: {
        name: user.userName,
        newPassword: newPassword,
      },
    });

    return {
      message: 'Password reset instructions have been sent to your email',
    };
  }

  private async hashPassword(password: string): Promise<string> {
    return argon.hash(password);
  }
}
