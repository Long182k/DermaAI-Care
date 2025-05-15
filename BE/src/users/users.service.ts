import { Injectable } from '@nestjs/common';
import { PrismaService } from 'src/prisma.service';
import { GetUserByKeywordDTO } from './dto/get-user.dto';
import {
  UpdateHashedRefreshTokenDTO,
  UpdateUserDto,
} from './dto/update-user.dto';
import { UserRepository } from './users.repository';

interface PaginationParams {
  page: number;
  limit: number;
}

@Injectable()
export class UsersService {
  constructor(
    private userRepository: UserRepository,
    private prisma: PrismaService,
  ) {}

  async updateHashedRefreshToken(
    updateHashedRefreshTokenDTO: UpdateHashedRefreshTokenDTO,
  ) {
    return await this.userRepository.updateHashedRefreshToken(
      updateHashedRefreshTokenDTO,
    );
  }

  async findDoctorByID(userId: string) {
    return await this.userRepository.findDoctorByID(userId);
  }

  async findAllDoctors() {
    return await this.userRepository.findAllDoctors();
  }

  async findOne(email: string) {
    return await this.userRepository.findUserByEmail(email);
  }

  async findUserByKeyword(keyword: GetUserByKeywordDTO) {
    return await this.userRepository.findUserByKeyword(keyword);
  }

  async editProfile(updateUserDto: UpdateUserDto, userId: string) {
    return await this.userRepository.update(userId, {
      ...updateUserDto,
      dateOfBirth: new Date(updateUserDto.dateOfBirth),
    });
  }

  async updateAvatar(id: string, avatarUrl: string) {
    return await this.userRepository.update(id, { avatarUrl });
  }

  async editUserNames(
    userId: string,
    dto: { firstName?: string; lastName?: string },
  ) {
    return this.prisma.user.update({
      where: { id: userId },
      data: {
        ...(dto.firstName !== undefined && { firstName: dto.firstName }),
        ...(dto.lastName !== undefined && { lastName: dto.lastName }),
      },
    });
  }

  async changeUserActive(userId: string, isActive: boolean) {
    return this.prisma.user.update({
      where: { id: userId },
      data: { isActive },
    });
  }
}
